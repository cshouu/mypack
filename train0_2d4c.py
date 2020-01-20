import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import FluidNet_2d4c

lamda_ndiv = 0.0
lamda_p = 0.15
lamda_v = 0.15
lamda_div = 0.7
PATH = './presnet_%.2f_%.2f_%.2f_%.2f_std_init1.pth' % (lamda_ndiv, lamda_p, lamda_v, lamda_div)
presnet = FluidNet_2d4c.PressureNet()
optimizer = optim.Adam(presnet.parameters())

writer = SummaryWriter('runs/fluidnet')

dt = 1.0
dx = 1/128
gravity = torch.tensor([0,-6e-4])
# train code
time_zero = time.time()
running_loss = 0.0
for epoch in range(320):
	# train
	itrainset = FluidNet_2d4c.VPGDdataset(epoch)
	for i in range(len(itrainset)):
		#with torch.autograd.set_detect_anomaly(True):
		isample = itrainset[i]
		# forward
		pres_1 = presnet(torch.unsqueeze(torch.cat([torch.unsqueeze(isample['th_vx'][:,:,0],0), torch.unsqueeze(isample['th_vx'][:,:,1],0), torch.unsqueeze(isample['th_px'],0), torch.unsqueeze(isample['th_gx'].float(),0)]),0))[0,0,:,:]
		vel_1 = FluidNet_2d4c.correctVel(isample['th_gx'], isample['th_vx'], pres_1)
		# loss
		loss = FluidNet_2d4c.objFunc(lamda_p=lamda_p, th_p=isample['th_p'], th_netp=pres_1, lamda_v=lamda_v, th_v=isample['th_v'], th_netv=vel_1, lamda_div=lamda_div)
		'''
		# forward for long term divergence
		##''''''''''''''''''' step 2 ''''''''''''''''''''
		dens_2 = FluidNet_2d4c.advectSemiLagrange_dens(dt=dt, vel=vel_1, dens=isample['th_dx'])
		vel_2_1 = FluidNet_2d4c.advectSemiLagrange_vel(dt=dt, vel=vel_1, velgrid=vel_1)
		vel_2_2 = FluidNet_2d4c.setWallBcs(isample['th_gx'], vel_2_1)		
		vel_2_3 = FluidNet_2d4c.addBuoyancy(flags= isample['th_gx'], density=dens_2, vel=vel_2_2, gravity=gravity, dt=dt, dx=dx, coefficient=1.0)
		vel_2_4 = FluidNet_2d4c.vorticityConfinement(vel=vel_2_3, flags=isample['th_gx'], strength=0.05)
		pres_2 = presnet(torch.unsqueeze(torch.cat([torch.unsqueeze(vel_2_4[:,:,0],0), torch.unsqueeze(vel_2_4[:,:,1],0), torch.unsqueeze(pres_1,0), torch.unsqueeze(isample['th_gx'].float(),0)]),0))[0,0,:,:]
		vel_2 = FluidNet_2d4c.correctVel(isample['th_gx'], vel_2_4, pres_2)
		##''''''''''''''''''' step 3 ''''''''''''''''''''
		dens_3 = FluidNet_2d4c.advectSemiLagrange_dens(dt=dt, vel=vel_2, dens=dens_2)
		vel_3_1 = FluidNet_2d4c.advectSemiLagrange_vel(dt=dt, vel=vel_2, velgrid=vel_2)
		vel_3_2 = FluidNet_2d4c.setWallBcs(isample['th_gx'], vel_3_1)		
		vel_3_3 = FluidNet_2d4c.addBuoyancy(flags= isample['th_gx'], density=dens_3, vel=vel_3_2, gravity=gravity, dt=dt, dx=dx, coefficient=1.0)
		vel_3_4 = FluidNet_2d4c.vorticityConfinement(vel=vel_3_3, flags=isample['th_gx'], strength=0.05)
		pres_3 = presnet(torch.unsqueeze(torch.cat([torch.unsqueeze(vel_3_4[:,:,0],0), torch.unsqueeze(vel_3_4[:,:,1],0), torch.unsqueeze(pres_2,0), torch.unsqueeze(isample['th_gx'].float(),0)]),0))[0,0,:,:]
		vel_3 = FluidNet_2d4c.correctVel(isample['th_gx'], vel_3_4, pres_3)
		##''''''''''''''''''' step 4 ''''''''''''''''''''
		dens_4 = FluidNet_2d4c.advectSemiLagrange_dens(dt=dt, vel=vel_3, dens=dens_3)
		vel_4_1 = FluidNet_2d4c.advectSemiLagrange_vel(dt=dt, vel=vel_3, velgrid=vel_3)
		vel_4_2 = FluidNet_2d4c.setWallBcs(isample['th_gx'], vel_4_1)		
		vel_4_3 = FluidNet_2d4c.addBuoyancy(flags= isample['th_gx'], density=dens_4, vel=vel_4_2, gravity=gravity, dt=dt, dx=dx, coefficient=1.0)
		vel_4_4 = FluidNet_2d4c.vorticityConfinement(vel=vel_4_3, flags=isample['th_gx'], strength=0.05)
		pres_4 = presnet(torch.unsqueeze(torch.cat([torch.unsqueeze(vel_4_4[:,:,0],0), torch.unsqueeze(vel_4_4[:,:,1],0), torch.unsqueeze(pres_3,0), torch.unsqueeze(isample['th_gx'].float(),0)]),0))[0,0,:,:]
		vel_4 = FluidNet_2d4c.correctVel(isample['th_gx'], vel_4_4, pres_4)
		##''''''''''''''''''' step 5 ''''''''''''''''''''
		dens_5 = FluidNet_2d4c.advectSemiLagrange_dens(dt=dt, vel=vel_4, dens=dens_4)
		vel_5_1 = FluidNet_2d4c.advectSemiLagrange_vel(dt=dt, vel=vel_4, velgrid=vel_4)
		vel_5_2 = FluidNet_2d4c.setWallBcs(isample['th_gx'], vel_5_1)		
		vel_5_3 = FluidNet_2d4c.addBuoyancy(flags= isample['th_gx'], density=dens_5, vel=vel_5_2, gravity=gravity, dt=dt, dx=dx, coefficient=1.0)
		vel_5_4 = FluidNet_2d4c.vorticityConfinement(vel=vel_5_3, flags=isample['th_gx'], strength=0.05)
		pres_5 = presnet(torch.unsqueeze(torch.cat([torch.unsqueeze(vel_5_4[:,:,0],0), torch.unsqueeze(vel_5_4[:,:,1],0), torch.unsqueeze(pres_4,0), torch.unsqueeze(isample['th_gx'].float(),0)]),0))[0,0,:,:]
		vel_5 = FluidNet_2d4c.correctVel(isample['th_gx'], vel_5_4, pres_5)
		# loss for long term divergence
		loss += lamda_ndiv * torch.norm(FluidNet_2d4c.divergenceOpVel(vel_5))
		'''
		# backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# accumulate loss
		running_loss += loss.item()
	# visiualization
	writer.add_scalar('training loss', running_loss/len(itrainset), epoch*len(itrainset))
	for name,layer in presnet.named_parameters():
		writer.add_histogram(name + '_grad', layer.grad.data, epoch)
		writer.add_histogram(name + '_weight', layer.data, epoch)
	# time
	time_all = time.time() - time_zero
	print('Dir: %03d  Time: %dm %ds' % (epoch+1, time_all//60,time_all%60))
	running_loss = 0.0
	# save presnet
	torch.save(presnet.state_dict(),PATH)




