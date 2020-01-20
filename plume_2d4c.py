#
# Simple example scene (hello world)
# Simulation of a buoyant smoke density plume (with noise texture as smoke source)
#

#import pdb; pdb.set_trace()

from manta import *

# solver params
res = 64
gs  = vec3(res, int(1.5*res), 1)
s   = FluidSolver(name='main', gridSize = gs, dim=2)

# prepare grids
flags    = s.create(FlagGrid)
vel      = s.create(MACGrid)
density  = s.create(RealGrid)
pressure = s.create(RealGrid)

flags.initDomain()
flags.fillGrid()

source = s.create(Cylinder, center=gs*vec3(0.5,0.1,0.5), radius=res*0.14, z=gs*vec3(0, 0.02, 0))

if (GUI):
	gui = Gui()
	gui.show()

#main loop
#'''
import numpy as np
import torch
import FluidNet_2d4c_no_long
np_dens = np.ones( [int(1.5*res), res], dtype=np.float32 )
np_pres = np.ones( [int(1.5*res), res], dtype=np.float32 )
np_vel = np.ones( [int(1.5*res), res, 3], dtype=np.float32 )
np_flag = np.ones( [int(1.5*res), res], dtype=np.int32 )

PATH = './lamda_0.00_1.00_0.00_0.00_bs_016.pth'
presnet = FluidNet_2d4c_no_long.PressureNet()
presnet.load_state_dict(torch.load(PATH))
presnet.eval()
#'''
for t in range(250):
	mantaMsg('\nFrame %i' % (s.frame))
	if t<100:
		source.applyToGrid(grid=density, value=1)
	'''
	copyGridToArrayReal(source=density,  target=np_dens)
	copyGridToArrayMAC(source=vel,  target=np_vel)
	th_vel = torch.from_numpy(np_vel)
	th_dens = torch.from_numpy(np_dens)
	th_dens = FluidNet_2d4c.advectSemiLagrange_dens(dt=s.timestep, vel=th_vel, dens=th_dens)
	np_dens = th_dens.detach().numpy()
	copyArrayToGridReal(source=np_dens,  target=density)
	'''
	advectSemiLagrange(flags=flags, vel=vel, grid=density, order=1)
	'''
	copyGridToArrayMAC(source=vel,  target=np_vel)
	th_vel = torch.from_numpy(np_vel)
	th_vel = FluidNet_2d4c.advectSemiLagrange_vel(dt=s.timestep, vel=th_vel, velgrid=th_vel)
	np_vel = th_vel.detach().numpy()
	copyArrayToGridMAC(source=np_vel,  target=vel)
	'''
	advectSemiLagrange(flags=flags, vel=vel, grid=vel)
	
	'''
	copyGridToArrayInt(source=flags,  target=np_flag)
	copyGridToArrayMAC(source=vel,  target=np_vel)
	th_vel = torch.from_numpy(np_vel)
	th_flag = torch.from_numpy(np_flag)
	FluidNet_2d4c.setWallBcs(th_flag, th_vel)
	np_vel = th_vel.detach().numpy()
	copyArrayToGridMAC(source=np_vel,  target=vel)
	'''
	setWallBcs(flags=flags, vel=vel)
	'''
	copyGridToArrayInt(source=flags,  target=np_flag)
	copyGridToArrayReal(source=density,  target=np_dens)
	copyGridToArrayMAC(source=vel,  target=np_vel)
	th_vel = torch.from_numpy(np_vel)
	th_dens = torch.from_numpy(np_dens)
	th_flag = torch.from_numpy(np_flag)
	FluidNet_2d4c.addBuoyancy(flags=th_flag, density=th_dens, vel=th_vel, gravity=torch.tensor([0,-6e-4]), dt=s.timestep, dx= 1.0/(1.5*res), coefficient=1.0)
	np_vel = th_vel.detach().numpy()
	copyArrayToGridMAC(source=np_vel,  target=vel)
	'''
	addBuoyancy(density=density, vel=vel, gravity=vec3(0,-6e-4,0), flags=flags)

	'''
	copyGridToArrayInt(source=flags,  target=np_flag)
	copyGridToArrayMAC(source=vel,  target=np_vel)
	th_vel = torch.from_numpy(np_vel)
	th_flag = torch.from_numpy(np_flag)
	FluidNet_2d4c.vorticityConfinement(vel=th_vel, flags=th_flag, strength=0.4)
	np_vel = th_vel.detach().numpy()
	copyArrayToGridMAC(source=np_vel,  target=vel)
	'''
	vorticityConfinement(vel=vel, flags=flags, strength=0.05)
	
	if t>200:
		copyGridToArrayInt(source=flags,  target=np_flag)
		copyGridToArrayReal(source=pressure,  target=np_pres)
		copyGridToArrayMAC(source=vel,  target=np_vel)
		th_vel = torch.from_numpy(np_vel)
		th_pres = torch.from_numpy(np_pres)
		th_flag = torch.from_numpy(np_flag)
		th_net_input = torch.unsqueeze(torch.cat([torch.unsqueeze(th_vel[:,:,0],0), torch.unsqueeze(th_vel[:,:,1],0), torch.unsqueeze(th_pres,0), torch.unsqueeze(th_flag.float(),0)]),0)
		th_vx = torch.unsqueeze(th_vel[:,:,:2], 0)
		th_gx = torch.unsqueeze(th_flag, 0)
		with torch.no_grad():
			th_pres, th_vel = presnet(th_net_input, th_vx, th_gx)
		th_pres = torch.squeeze(th_pres)
		th_vel = torch.squeeze(th_vel)

		np_pres = th_pres.numpy()
		np_vel[:,:,:2] = th_vel.numpy()
		copyArrayToGridReal(source=np_pres,  target=pressure)
		copyArrayToGridMAC(source=np_vel,  target=vel)
	else:
		solvePressure( flags=flags, vel=vel, pressure=pressure )

	s.step()

