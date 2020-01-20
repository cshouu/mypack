from collections import namedtuple
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

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

class VPGDdataset(Dataset):
	def __init__(self,isTrainset=True):
		if isTrainset:
			self.dataset = 'tr'
		else:
			self.dataset = 'te'
	def __len__(self):
		return 64*320
	def __getitem__(self,idx):
		idir = idx // 64
		ifile = (idx % 64) * 4
		th_vx,th_px,th_gx,th_dx = readbin('./data/%s/%06d/%06d_divergent.bin' % (self.dataset,idir,ifile))
		th_v,th_p,_1,_2 = readbin('./data/%s/%06d/%06d.bin' % (self.dataset,idir,ifile))
		th_net_input = torch.cat([torch.unsqueeze(th_vx[:,:,0],0), torch.unsqueeze(th_vx[:,:,1],0), torch.unsqueeze(th_px,0), torch.unsqueeze(th_gx.float(),0)])
		return {'th_net_input':th_net_input, 'th_vx':th_vx,
'th_gx':th_gx, 'th_dx':th_dx, 'th_v':th_v, 'th_p':th_p}


class PressureNet(nn.Module):
	def __init__(self):
		super(PressureNet,self).__init__()
		self.conv1 = nn.Conv2d(in_channels=4,out_channels=16,kernel_size=(3,3),padding=(1,1))
		self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=(1,1))
		self.conv3 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=(1,1))
		self.conv4 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=(1,1))
		self.conv5 = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=(1,1),padding=(0,0))

	def forward(self, x, vel_div, flags):
		# scale
		std = torch.std(x[:,:2,:,:].view(x.shape[0],-1))
		scale = torch.clamp(std, 0.00001, float('inf'))
		x[:,:3,:,:] /= scale
		# forward
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		p_pred = torch.squeeze(F.relu(self.conv5(x)), 1)
		# reverse scale
		v_pred = self.correctVel(flags, vel_div, p_pred)
		p_pred *= scale
		v_pred *= scale
		return p_pred, v_pred

	def correctVel(self, flags, vel, pressure):
		vel_corrected = vel.clone()
		vel_corrected[:,1:-1,1:-1,0] -= (flags[:,1:-1,1:-1]&1!=0) * (flags[:,1:-1,:-2]&1!=0) * (pressure[:,1:-1,1:-1] - pressure[:,1:-1,:-2])
		vel_corrected[:,1:-1,1:-1,1] -= (flags[:,1:-1,1:-1]&1!=0) * (flags[:,:-2,1:-1]&1!=0) * (pressure[:,1:-1,1:-1] - pressure[:,:-2,1:-1])
		vel_corrected[:,1:-1,1:-1,0] -= (flags[:,1:-1,1:-1]&1!=0) * (flags[:,1:-1,:-2]&4!=0) * pressure[:,1:-1,1:-1]
		vel_corrected[:,1:-1,1:-1,1] -= (flags[:,1:-1,1:-1]&1!=0) * (flags[:,:-2,1:-1]&4!=0) * pressure[:,1:-1,1:-1]
		vel_corrected[:,1:-1,1:-1,0] = ((flags[:,1:-1,1:-1]&4!=0)*~(flags[:,1:-1,1:-1]&16!=0)) * ((flags[:,1:-1,:-2]&1!=0) * (vel_corrected[:,1:-1,1:-1,0] + pressure[:,1:-1,:-2]) + (flags[:,1:-1,:-2]&1==0) * 0.0) + ((flags[:,1:-1,1:-1]&4==0)+(flags[:,1:-1,1:-1]&16!=0)>0) * vel_corrected[:,1:-1,1:-1,0]
		vel_corrected[:,1:-1,1:-1,1] = ((flags[:,1:-1,1:-1]&4!=0)*~(flags[:,1:-1,1:-1]&16!=0)) * ((flags[:,:-2,1:-1]&1!=0) * (vel_corrected[:,1:-1,1:-1,1] + pressure[:,:-2,1:-1]) + (flags[:,:-2,1:-1]&1==0) * 0.0) + ((flags[:,1:-1,1:-1]&4==0)+(flags[:,1:-1,1:-1]&16!=0)>0) * vel_corrected[:,1:-1,1:-1,1]
		return vel_corrected


class LossFunc(nn.Module):
	def __init__(self):
		super(LossFunc, self).__init__()
		self.divergenceOpVel = DivergenceOpVel()

	def forward(self, lamda_p, lamda_v, lamda_div, th_p, th_v, th_netp, th_netv):
		error_p = th_p - th_netp
		error_v = th_v - th_netv
		error_div = self.divergenceOpVel(th_netv)
		return (lamda_p * torch.sum(error_p * error_p) + lamda_v * torch.sum(error_v * error_v) + lamda_div * torch.sum(error_div * error_div)) / th_p.shape[0]


class DivergenceOpVel(nn.Module):
	def __init__(self):
		super(DivergenceOpVel, self).__init__()

	def forward(self, vel):
		div = torch.zeros_like(vel[:,:,:,0])
		div[:, 1:-1, 1:-1] = (vel[:, 1:-1, 2:, 0] - vel[:, 1:-1, 1:-1, 0]) + (vel[:, 2:, 1:-1, 1] - vel[:, 1:-1, 1:-1, 1])
		return div

