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
	def __init__(self,ratio_trainset=0.8,radio_validset=0.2,trainset=True):
		self.ratio_trainset = ratio_trainset
		self.radio_validset = radio_validset
		self.trainset = trainset
	def __len__(self):
		if self.trainset:
			return int(64*320 * self.ratio_trainset)
		else:
			return int(64*320 * self.radio_validset)
	def __getitem__(self,idx):
		if self.trainset:
			pass
		else:
			idx = idx + 64*320*self.ratio_trainset
		idir = idx // 64
		ifile = (idx % 64) * 4
		th_vx,th_px,th_gx,th_dx = readbin('./data/tr/%06d/%06d_divergent.bin' % (idir,ifile))
		th_v,th_p,_1,_2 = readbin('./data/tr/%06d/%06d.bin' % (idir,ifile))
		th_net_input = torch.cat([torch.unsqueeze(th_vx[:,:,0],0), torch.unsqueeze(th_vx[:,:,1],0), torch.unsqueeze(th_px,0), torch.unsqueeze(th_gx.float(),0)])
		return {'th_net_input':th_net_input, 'th_vx':th_vx,
'th_gx':th_gx, 'th_dx':th_dx, 'th_v':th_v, 'th_p':th_p}

# pytorch pressure Net
class PressureNet(nn.Module):
	def __init__(self):
		super(PressureNet,self).__init__()
		self.conv1 = nn.Conv2d(in_channels=4,out_channels=16,kernel_size=(3,3),padding=(1,1))
		self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=(1,1))
		self.conv3 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=(1,1))
		self.conv4 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=(1,1))
		self.conv5 = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=(1,1),padding=(0,0))

	def forward(self, x, vel_div, flags):
		std = torch.std(x[:,:2,:,:].view(x.shape[0],-1))
		scale = torch.clamp(std,0.00001,float('inf'))

		x[:,:3,:,:] /= scale
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		p_pred = F.relu(self.conv5(x))

		v_pred = self.correctVel(flags, vel_div, p_pred)

		p_pred *= scale
		v_pred *= scale
		return p_pred, v_pred

	def correctVel(flags, vel, pressure):
		vel_corrected = vel.clone()

		vel_corrected[:,1:-1,1:-1,0] -= (flags[:,1:-1,1:-1]&1!=0) * (flags[:,1:-1,:-2]&1!=0) * (pressure[:,1:-1,1:-1] - pressure[:,1:-1,:-2])
		vel_corrected[:,1:-1,1:-1,1] -= (flags[:,1:-1,1:-1]&1!=0) * (flags[:,:-2,1:-1]&1!=0) * (pressure[:,1:-1,1:-1] - pressure[:,:-2,1:-1])

		vel_corrected[:,1:-1,1:-1,0] -= (flags[:,1:-1,1:-1]&1!=0) * (flags[:,1:-1,:-2]&4!=0) * pressure[:,1:-1,1:-1]
		vel_corrected[:,1:-1,1:-1,1] -= (flags[:,1:-1,1:-1]&1!=0) * (flags[:,:-2,1:-1]&4!=0) * pressure[:,1:-1,1:-1]

		vel_corrected[:,1:-1,1:-1,0] = ((flags[:,1:-1,1:-1]&4!=0)*~(flags[:,1:-1,1:-1]&16!=0)) * ((flags[:,1:-1,:-2]&1!=0) * (vel_corrected[:,1:-1,1:-1,0] + pressure[:,1:-1,:-2]) + (flags[:,1:-1,:-2]&1==0) * 0.0) + ((flags[:,1:-1,1:-1]&4==0)+(flags[:,1:-1,1:-1]&16!=0)>0) * vel_corrected[:,1:-1,1:-1,0]
		vel_corrected[:,1:-1,1:-1,1] = ((flags[:,1:-1,1:-1]&4!=0)*~(flags[:,1:-1,1:-1]&16!=0)) * ((flags[:,:-2,1:-1]&1!=0) * (vel_corrected[:,1:-1,1:-1,1] + pressure[:,:-2,1:-1]) + (flags[:,:-2,1:-1]&1==0) * 0.0) + ((flags[:,1:-1,1:-1]&4==0)+(flags[:,1:-1,1:-1]&16!=0)>0) * vel_corrected[:,1:-1,1:-1,1]
		return vel_corrected

class lossFunc(nn.Module):
	def __init__(self):
		super(lossFunc, self).__init__()
	def forward(self, th_p, th_netp, th_v, th_netv, lamda_p, lamda_v, lamda_div):
		error_p = th_p - th_netp
		error_v = th_v - th_netv
		error_div = divergenceOpVel(th_netv)
		return (lamda_p * torch.sum(error_p * error_p) + lamda_v * torch.sum(error_v * error_v) + lamda_div * torch.sum(error_div * error_div)) / th_p.shape[0]

class divergenceOpVel(nn.Module):
	def __init__(self):
		super(divergenceOpVel, self).__init__()
	def forward(self, vel):
		div = torch.zeros(vel.shape[0], vel.shape[1], vel.shape[2], dtype=torch.float)
		div[:, 1:-1, 1:-1] = (vel[:, 1:-1, 2:, 0] - vel[:, 1:-1, 1:-1, 0]) + (vel[:, 2:, 1:-1, 1] - vel[:, 1:-1, 1:-1, 1])
		return div

############## functions for long divergence ############

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


