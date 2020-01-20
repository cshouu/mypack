from collections import namedtuple
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

'''
! types of cells, in/outflow can be combined, e.g., TypeFluid|TypeInflow
CellType { 
	TypeNone     = 0,
	TypeFluid    = 1,
	TypeObstacle = 2,
	TypeEmpty    = 4,
	TypeInflow   = 8,
	TypeOutflow  = 16,
	TypeOpen     = 32,
	TypeStick    = 64,
	// internal use only, for fast marching
	TypeReserved = 256,
	// 2^10 - 2^14 reserved for moving obstacles
}
'''
########### about net without long divergence ############
def readbin(filename):
	with open(filename,'rb') as bytestream:
		Header = namedtuple('Header','transpose nx ny nz is_3d')
		header = Header._make(struct.unpack('iiiii',bytestream.read(20)))
		size4 = header.nx * header.ny * 4
		np_vel = np.zeros([header.ny,header.nx,2],dtype=np.float32)
		np_vel[:,:,0] = np.frombuffer(bytestream.read(size4),dtype='float32').reshape(header.ny,header.nx)
		np_vel[:,:,1] = np.frombuffer(bytestream.read(size4),dtype='float32').reshape(header.ny,header.nx)
		np_pres = np.frombuffer(bytestream.read(size4),dtype='float32').reshape(header.ny,header.nx)
		np_flag = np.frombuffer(bytestream.read(size4),dtype='int32').reshape(header.ny,header.nx)
		np_dens = np.frombuffer(bytestream.read(size4),dtype='float32').reshape(header.ny,header.nx)
		th_vel = torch.from_numpy(np_vel)
		th_pres = torch.from_numpy(np_pres)
		th_flag = torch.from_numpy(np_flag)
		th_dens = torch.from_numpy(np_dens)
		return th_vel,th_pres,th_flag,th_dens

# pytorch dataset
class VPGDdataset(Dataset):
	def __init__(self,idir):
		self.idir = idir
	def __len__(self):
		# 64 examples in each dir
		return 64
	def __getitem__(self,idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		idx *=  4
		th_vx,th_px,th_gx,th_dx = readbin('./data/tr/%06d/%06d_divergent.bin' % (self.idir,idx))
		th_v,th_p,_1,_2 = readbin('./data/tr/%06d/%06d.bin' % (self.idir,idx))
		return {'th_vx':th_vx, 'th_px':th_px,
'th_gx':th_gx, 'th_dx':th_dx, 'th_v':th_v, 'th_p':th_p}

# pytorch pressure Net
class PressureNet(nn.Module):
	def __init__(self):
		super(PressureNet,self).__init__()
		self.conv1 = nn.Conv2d(in_channels=4,out_channels=16,kernel_size=(3,3),padding=(1,1))
		self.maxPool1 = nn.MaxPool2d(kernel_size=(2,2))
		self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=(1,1))
		self.maxPool2 = nn.MaxPool2d(kernel_size=(2,2))
		self.conv3 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=(1,1))
		self.conv4 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=(1,1))
		self.conv5 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(1,1))
		self.conv6 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1))
		self.upconv = nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=(3,3),stride=(4,4),padding=(1,1),output_padding=(3,3))
		self.initialize_weights()
	def forward(self,x):
		std = torch.std(x[0,2,:,:]) + 1e-05
		x[0,:3,:,:] /= std
		x = F.relu(self.conv1(x))
		x = self.maxPool1(x)
		x = F.relu(self.conv2(x))
		x = self.maxPool2(x)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x = F.relu(self.conv6(x))
		x = F.relu(self.upconv(x))
		x *= std
		return x
	def initialize_weights(self):
		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
				if m.bias is not None:
					m.bias.data.zero_()

# pytorch objFunc()
def objFunc(lamda_p, th_p, th_netp, lamda_v, th_v, th_netv, lamda_div):
	return lamda_p * torch.norm(th_p - th_netp) + lamda_v * torch.norm(th_v - th_netv) + lamda_div * torch.norm(divergenceOpVel(th_netv))

# mantaflow DivergenceOpMAC() in commonkernels.h KERNEL(bnd=1)
def divergenceOpVel(vel):
	div = torch.zeros(vel.shape[0], vel.shape[1], dtype=torch.float)
	div[1:-1,1:-1] = (vel[1:-1,2:,0] - vel[1:-1,1:-1,0]) + (vel[2:,1:-1,1] - vel[1:-1,1:-1,1])
	return div

# mantaflow correctVel()
# KERNEL(bnd = 1)
def correctVel(flags, vel, pressure):
	vel_corrected = vel.clone()
	vel_corrected[1:-1,1:-1,0] -= (flags[1:-1,1:-1]&1!=0) * (flags[1:-1,:-2]&1!=0) * (pressure[1:-1,1:-1] - pressure[1:-1,:-2])
	vel_corrected[1:-1,1:-1,1] -= (flags[1:-1,1:-1]&1!=0) * (flags[:-2,1:-1]&1!=0) * (pressure[1:-1,1:-1] - pressure[:-2,1:-1])
	vel_corrected[1:-1,1:-1,0] -= (flags[1:-1,1:-1]&1!=0) * (flags[1:-1,:-2]&4!=0) * pressure[1:-1,1:-1]
	vel_corrected[1:-1,1:-1,1] -= (flags[1:-1,1:-1]&1!=0) * (flags[:-2,1:-1]&4!=0) * pressure[1:-1,1:-1]
	vel_corrected[1:-1,1:-1,0] = ((flags[1:-1,1:-1]&4!=0)*~(flags[1:-1,1:-1]&16!=0)) * ((flags[1:-1,:-2]&1!=0) * (vel_corrected[1:-1,1:-1,0] + pressure[1:-1,:-2]) + (flags[1:-1,:-2]&1==0) * 0.0) + ((flags[1:-1,1:-1]&4==0)+(flags[1:-1,1:-1]&16!=0)>0) * vel_corrected[1:-1,1:-1,0]
	vel_corrected[1:-1,1:-1,1] = ((flags[1:-1,1:-1]&4!=0)*~(flags[1:-1,1:-1]&16!=0)) * ((flags[:-2,1:-1]&1!=0) * (vel_corrected[1:-1,1:-1,1] + pressure[:-2,1:-1]) + (flags[:-2,1:-1]&1==0) * 0.0) + ((flags[1:-1,1:-1]&4==0)+(flags[1:-1,1:-1]&16!=0)>0) * vel_corrected[1:-1,1:-1,1]
	return vel_corrected

#######################################################

################## advectSemiLagrange_dens ################

# mantaflow intepol()
# used by advectSemiLagrange_dens()(KERNEL(bnd=1))
# orderSpace:1
def intepol(src, pos):
	px, py = pos[:,:,0] - 0.5, pos[:,:,1] - 0.5
	xi, yi = px.int(), py.int()
	s1 = px - xi
	s0 = 1.0 - s1
	t1 = py - yi
	t0 = 1.0 - t1
	# clamp to border
	xi = (px < 0.0) * 0 + ~(px < 0.0) * xi
	s0 = (px < 0.0) * 1.0 + ~(px < 0.0) * s0
	s1 = (px < 0.0) * 0.0 + ~(px < 0.0) * s1
	yi = (py < 0.0) * 0 + ~(py < 0.0) * yi
	t0 = (py < 0.0) * 1.0 + ~(py < 0.0) * t0
	t1 = (py < 0.0) * 0.0 + ~(py < 0.0) * t1
	xi = (px >= src.shape[1]-1)*(src.shape[1]-2) + ~(px >= src.shape[1]-1)*xi
	s0 = (px >= src.shape[1]-1)*0.0 + ~(px >= src.shape[1]-1)*s0
	s1 = (px >= src.shape[1]-1)*1.0 + ~(px >= src.shape[1]-1)*s1
	yi = (py >= src.shape[0]-1)*(src.shape[0]-2) + ~(py >= src.shape[0]-1)*yi
	t0 = (py >= src.shape[0]-1)*0.0 + ~(py >= src.shape[0]-1)*t0
	t1 = (py >= src.shape[0]-1)*1.0 + ~(py >= src.shape[0]-1)*t1
	return (src[yi,xi] * t0 + src[yi+1,xi] * t1) * s0 + (src[yi,xi+1] * t0 + src[yi+1,xi+1] * t1) * s1

# mantaflow getCentered() 
# used by advectSemiLagrange_dens()(KERNEL(bnd=1))
def getCentered(vel):
	vel_centered = torch.zeros_like(vel)
	vel_centered[1:-1, 1:-1, 0] = vel[1:-1, 1:-1, 0] + vel[1:-1, 2:, 0]
	vel_centered[1:-1, 1:-1, 1] = vel[1:-1, 1:-1, 1] + vel[2:, 1:-1, 1]
	return vel_centered * 0.5

# mantaflow advectSemiLagrange_dens()
# advect order:1
def advectSemiLagrange_dens(dt, vel, dens, order = 1, strength = 1.0, orderSpace = 1, openBounds = False, boundaryWidth = 1, clampMode = 2):
	dens_next = torch.zeros_like(dens)
	pos = torch.zeros(dens.shape[0], dens.shape[1], 2)
	pos[1:-1,1:-1,0] = torch.arange(1, dens.shape[1]-1, dtype=torch.float) + 0.5
	pos[1:-1,1:-1,1] = torch.arange(1, dens.shape[0]-1, dtype=torch.float).view(-1,1) + 0.5
	pos[1:-1,1:-1,:] = pos[1:-1,1:-1,:] - getCentered(vel)[1:-1,1:-1,:2] * dt
	dens_next[1:-1,1:-1] = intepol(dens, pos)[1:-1,1:-1]
	return dens_next

################################################################

################### advectSemiLagrange_vel #####################

# mantaflow interpolComponent()
# used by SemiLagrangeMAC()(KERNEL(bnd=1))
# orderSpace:1
def interpolComponent(src, pos, orderSpace, c):
	px, py = pos[:,:,0] - 0.5, pos[:,:,1] - 0.5
	xi, yi = px.int(), py.int()
	s1 = px - xi
	s0 = 1.0 - s1
	t1 = py - yi
	t0 = 1.0 - t1
	# clamp to border
	xi = (px < 0.0) * 0 + ~(px < 0.0) * xi
	s0 = (px < 0.0) * 1.0 + ~(px < 0.0) * s0
	s1 = (px < 0.0) * 0.0 + ~(px < 0.0) * s1
	yi = (py < 0.0) * 0 + ~(py < 0.0) * yi
	t0 = (py < 0.0) * 1.0 + ~(py < 0.0) * t0
	t1 = (py < 0.0) * 0.0 + ~(py < 0.0) * t1
	xi = (px >= src.shape[1]-1)*(src.shape[1]-2) + ~(px >= src.shape[1]-1)*xi
	s0 = (px >= src.shape[1]-1)*0.0 + ~(px >= src.shape[1]-1)*s0
	s1 = (px >= src.shape[1]-1)*1.0 + ~(px >= src.shape[1]-1)*s1
	yi = (py >= src.shape[0]-1)*(src.shape[0]-2) + ~(py >= src.shape[0]-1)*yi
	t0 = (py >= src.shape[0]-1)*0.0 + ~(py >= src.shape[0]-1)*t0
	t1 = (py >= src.shape[0]-1)*1.0 + ~(py >= src.shape[0]-1)*t1
	return (src[yi,xi,c] * t0 + src[yi+1,xi,c] * t1) * s0 + (src[yi,xi+1,c] * t0 + src[yi+1,xi+1,c] * t1) * s1

# mantaflow getAtMACX()
# used by SemiLagrangeMAC() KERNEL(bnd=1)
def getAtMACX(vel):
	vel_macx = torch.zeros_like(vel)
	vel_macx[1:-1,1:-1,0] = vel[1:-1,1:-1,0]
	vel_macx[1:-1,1:-1,1] = (vel[1:-1,1:-1,1] + vel[1:-1,:-2,1] + vel[2:,1:-1,1] + vel[2:,:-2,1])*0.25
	return vel_macx

# mantaflow getAtMACY()
# used by SemiLagrangeMAC() KERNEL(bnd=1)
def getAtMACY(vel):
	vel_macy = torch.zeros_like(vel)
	vel_macy[1:-1,1:-1,0] = (vel[1:-1,1:-1,0] + vel[:-2,1:-1,0] + vel[1:-1,2:,0] + vel[:-2,2:,0])*0.25
	vel_macy[1:-1,1:-1,1] = vel[1:-1,1:-1,1]
	return vel_macy

# mantaflow advectSemiLagrange_vel()
# advect order:1
def advectSemiLagrange_vel(dt, vel, velgrid, order = 1, strength = 1.0, orderSpace = 1, openBounds = False, boundaryWidth = 1, clampMode = 2):
	vel_next = torch.zeros_like(velgrid)
	pos = torch.zeros(velgrid.shape[0], velgrid.shape[1], 2)
	pos[1:-1,1:-1,0] = torch.arange(1, velgrid.shape[1]-1, dtype=torch.float) + 0.5
	pos[1:-1,1:-1,1] = torch.arange(1, velgrid.shape[0]-1, dtype=torch.float).view(-1,1) + 0.5
	xpos = pos - getAtMACX(vel)[:,:,:2] * dt
	vel_next[1:-1,1:-1,0] = interpolComponent(velgrid, xpos, orderSpace, 0)[1:-1,1:-1]
	ypos = pos - getAtMACY(vel)[:,:,:2] * dt
	vel_next[1:-1,1:-1,1] = interpolComponent(velgrid, ypos, orderSpace, 1)[1:-1,1:-1]
	return vel_next

##############################################################

# mantaflow setWallBcs()
# set obstacle boundary conditions
# ! set no-stick wall boundary condition between ob/fl and ob/ob cells
def setWallBcs(flags, vel):
	vel_next = vel.clone()
	vel_next[:,1:,0] = (((flags[:,1:]&1!=0)+(flags[:,1:]&2!=0))>0) * (flags[:,:-1]&2!=0) * 0.0 + ((flags[:,1:]&1==0) * (flags[:,1:]&2==0)+(flags[:,:-1]&2==0)>0) * vel_next[:,1:,0]
	vel_next[:,1:,0] = (flags[:,1:]&2!=0)*(flags[:,:-1]&1!=0) * 0.0 + (((flags[:,1:]&2==0)+(flags[:,:-1]&1==0))>0) * vel_next[:,1:,0]
	vel_next[1:,:,1] = (((flags[1:,:]&1!=0)+(flags[1:,:]&2!=0))>0) * (flags[:-1,:]&2!=0) * 0.0 + ((flags[1:,:]&1==0) * (flags[1:,:]&2==0)+(flags[:-1,:]&2==0)>0) * vel_next[1:,:,1]
	vel_next[1:,:,1] = (flags[1:,:]&2!=0)*(flags[:-1,:]&1!=0) * 0.0 + (((flags[1:,:]&2==0)+(flags[:-1,:]&1==0))>0) * vel_next[1:,:,1]
	vel_next[:,1:,1] = (flags[:,1:]&1!=0)*(flags[:,:-1]&64!=0) * 0.0 + (((flags[:,1:]&1==0)+(flags[:,:-1]&64==0))>0) * vel_next[:,1:,1]
	vel_next[:,:-1,1] = (flags[:,:-1]&1!=0)*(flags[:,1:]&64!=0) * 0.0 + (((flags[:,:-1]&1==0)+(flags[:,1:]&64==0))>0) * vel_next[:,:-1,1]
	vel_next[1:,:,0] = (flags[1:,:]&1!=0)*(flags[:-1,:]&64!=0) * 0.0 + (((flags[1:,:]&1==0)+(flags[:-1,:]&64==0))>0) * vel_next[1:,:,0]
	vel_next[:-1,:,0] = (flags[:-1,:]&1!=0)*(flags[1:,:]&64!=0) * 0.0 + (((flags[:-1,:]&1==0)+(flags[1:,:]&64==0))>0) * vel_next[:-1,:,0]
	return vel_next

# mantaflow addBuoyancy()
# ! add Buoyancy force based on factor (e.g. smoke density)
# KERNEL(bnd=1)
def addBuoyancy(flags, density, vel, gravity, dt, dx, coefficient=1.0):
	vel_next = vel.clone()
	strength = -gravity * dt / dx * coefficient
	vel_next[1:-1,1:-1,0] += (flags[1:-1,1:-1]&1!=0) * (flags[1:-1,:-2]&1!=0) * (0.5 * strength[0] * (density[1:-1,1:-1] + density[1:-1,:-2]))
	vel_next[1:-1,1:-1,1] += (flags[1:-1,1:-1]&1!=0) * (flags[:-2,1:-1]&1!=0) * (0.5 * strength[1] * (density[1:-1,1:-1] + density[:-2,1:-1]))
	return vel_next

# mantaflow KERNEL(bnd=1)
def vorticityConfinement(vel, flags, strength):
	velCenter = getCentered(vel)
	curl = torch.zeros(flags.shape[0], flags.shape[1], dtype=torch.float)
	curl[1:-1,1:-1] = ((velCenter[1:-1,2:,1] - velCenter[1:-1,:-2,1]) - (velCenter[2:,1:-1,0] - velCenter[:-2,1:-1,0])) * 0.5
	norm = torch.abs(curl)
	grad = torch.zeros(flags.shape[0], flags.shape[1], 2, dtype=torch.float)
	grad[1:-1,1:-1,0] = 0.5 * (norm[1:-1,2:] - norm[1:-1,:-2])
	grad[1:-1,1:-1,1] = 0.5 * (norm[2:,1:-1] - norm[:-2,1:-1])
	grad_normalized = torch.zeros_like(grad)
	grad_normalized[1:-1,1:-1,:] = F.normalize(grad[1:-1,1:-1,:], dim=0)
	force = torch.zeros(flags.shape[0], flags.shape[1], 2, dtype=torch.float)
	force[1:-1,1:-1,0] = grad_normalized[1:-1,1:-1,1] * curl[1:-1,1:-1] * strength
	force[1:-1,1:-1,1] = -grad_normalized[1:-1,1:-1,0] * curl[1:-1,1:-1] * strength
	forceX = torch.zeros(flags.shape[0], flags.shape[1], dtype=torch.float)
	forceY = torch.zeros(flags.shape[0], flags.shape[1], dtype=torch.float)
	forceX[1:-1,1:-1] = 0.5 * (force[1:-1,:-2,0] + force[1:-1,1:-1,0])
	forceY[1:-1,1:-1] = 0.5 * (force[:-2,1:-1,1] + force[1:-1,1:-1,1])
	vel_next = vel.clone()
	vel_next[1:-1,1:-1,0] = ((((flags[1:-1,1:-1]&1!=0) + (flags[1:-1,1:-1]&4!=0)>0) * (flags[1:-1,:-2]&1!=0)) + ((flags[1:-1,1:-1]&1!=0) * (flags[1:-1,:-2]&4!=0))>0) * (vel_next[1:-1,1:-1,0] + forceX[1:-1,1:-1]) + ((flags[1:-1,1:-1]&1==0) * (flags[1:-1,1:-1]&4==0) + (flags[1:-1,:-2]&1==0) * ((flags[1:-1,1:-1]&1==0) + (flags[1:-1,:-2]&4==0)>0)>0) * vel_next[1:-1,1:-1,0]
	vel_next[1:-1,1:-1,1] = (((flags[1:-1,1:-1]&1!=0) + (flags[1:-1,1:-1]&4!=0)>0) * (flags[:-2,1:-1]&1!=0) + ((flags[1:-1,1:-1]&1!=0) * (flags[:-2,1:-1]&4!=0))>0) * (vel_next[1:-1,1:-1,1] + forceY[1:-1,1:-1]) + ((flags[1:-1,1:-1]&1==0) * (flags[1:-1,1:-1]&4==0) + (flags[:-2,1:-1]&1==0) * ((flags[1:-1,1:-1]&1==0) + (flags[:-2,1:-1]&4==0)>0)>0) * vel_next[1:-1,1:-1,1]
	return vel_next


