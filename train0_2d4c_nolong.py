import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import FluidNet_2d4c_no_long
from torch.utils.tensorboard import SummaryWriter
import math
import time

time_zero = time.time()
time_last = time_zero
writer = SummaryWriter('runs')

# gpu
use_gpu = torch.cuda.is_available()

# config
train_batch_size = 16
test_batch_size = 16
train_shuffle = True
test_shuffle = False
train_num_workers = 2
test_num_workers = 2

epoch = 0
normalizeInputChan = torch.tensor(0) # 0:'UDiv' or 1:'pDiv' or 2:'div'
# normalizeInputFunc = 'std'
normalizeInputThrehsold = torch.tensor(0.00001)

# data
trainset = FluidNet_2d4c_no_long.VPGDdataset(isTrainset=True)
trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=train_shuffle, num_workers=train_num_workers)
testset = FluidNet_2d4c_no_long.VPGDdataset(isTrainset=False)
testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=test_shuffle, num_workers=test_num_workers)

# model
presnet = FluidNet_2d4c_no_long.PressureNet()




maxEpochs = 50 #5000
optimal_epoch = 0
optimal_test_loss = float('inf')


lamda_ndiv = 0.0
lamda_div = torch.tensor(1.0)
lamda_p = torch.tensor(0.0)
lamda_v = torch.tensor(0.0)
PATH = './lamda_%.2f_%.2f_%.2f_%.2f_bs_%03d.pth' % (lamda_ndiv, lamda_div, lamda_p, lamda_v, train_batch_size)
PATH_LAST = './lamda_%.2f_%.2f_%.2f_%.2f_bs_%03d_last.pth' % (lamda_ndiv, lamda_div, lamda_p, lamda_v, train_batch_size)

dt = 0.1
dx = 1/128
gravity = torch.tensor([0, -6e-4])


train_batch_total = math.ceil(len(trainset) / train_batch_size)
test_batch_total = math.ceil(len(testset) / test_batch_size)


lossFunc = FluidNet_2d4c_no_long.LossFunc()
if use_gpu:
	presnet = presnet.cuda()
	lossFunc = lossFunc.cuda()
optimizer = optim.Adam(presnet.parameters())

while epoch < maxEpochs:
	time_start = time.time()
	print('Epoch: %02d  Time: %dm  Delta_time: %dm ...' % (epoch, (time_start-time_zero)//60, (time_start-time_last)//60))
	time_last = time_start
	# Train code
	train_loss = 0.0
	for data_batched in trainloader:
		if use_gpu:
			data_batched['th_net_input'] = data_batched['th_net_input'].cuda()
			data_batched['th_vx'] = data_batched['th_vx'].cuda()
			data_batched['th_gx'] = data_batched['th_gx'].cuda()
			lamda_p = lamda_p.cuda()
			lamda_v = lamda_v.cuda()
			lamda_div = lamda_div.cuda()
			data_batched['th_p'] = data_batched['th_p'].cuda()
			data_batched['th_v'] = data_batched['th_v'].cuda()
		optimizer.zero_grad()
		pres, vel = presnet(data_batched['th_net_input'], data_batched['th_vx'], data_batched['th_gx'], normalizeInputChan=normalizeInputChan, normalizeInputThrehsold=normalizeInputThrehsold)
		loss = lossFunc(lamda_p=lamda_p, lamda_v=lamda_v, lamda_div=lamda_div, th_p=data_batched['th_p'], th_v=data_batched['th_v'], th_netp=pres, th_netv=vel)
		loss.backward()
		optimizer.step()
		if use_gpu:
			loss = loss.cpu()
		train_loss += loss.item()
	train_loss /= train_batch_total
	# test code
	test_loss = 0.0
	with torch.no_grad():
		for data_batched in testloader:
			if use_gpu:
				data_batched['th_net_input'] = data_batched['th_net_input'].cuda()
				data_batched['th_vx'] = data_batched['th_vx'].cuda()
				data_batched['th_gx'] = data_batched['th_gx'].cuda()
				lamda_p = lamda_p.cuda()
				lamda_v = lamda_v.cuda()
				lamda_div = lamda_div.cuda()
				data_batched['th_p'] = data_batched['th_p'].cuda()
				data_batched['th_v'] = data_batched['th_v'].cuda()
			pres, vel = presnet(data_batched['th_net_input'], data_batched['th_vx'], data_batched['th_gx'], normalizeInputChan=normalizeInputChan, normalizeInputThrehsold=normalizeInputThrehsold)
			loss = lossFunc(lamda_p=lamda_p, lamda_v=lamda_v, lamda_div=lamda_div, th_p=data_batched['th_p'], th_v=data_batched['th_v'], th_netp=pres, th_netv=vel)
			if use_gpu:
				loss = loss.cpu()
			test_loss += loss.item()
		test_loss /= test_batch_total
	# save model
	if test_loss < optimal_test_loss:
		if use_gpu:
			presnet = presnet.cpu()
		torch.save(presnet.state_dict(), PATH)
		if use_gpu:
			presnet = presnet.cuda()
		optimal_epoch = epoch
		optimal_test_loss = test_loss
	if epoch == maxEpochs:
		if use_gpu:
			presnet = presnet.cpu()
		torch.save(presnet.state_dict(), PATH_LAST)
	# visiualization
	writer.add_scalar('train loss', train_loss, epoch)
	writer.add_scalar('test loss', test_loss, epoch)
	for name,layer in presnet.named_parameters():
		writer.add_histogram(name + '_grad', layer.grad.data, epoch)
		writer.add_histogram(name + '_weight', layer.data, epoch)
	epoch += 1



