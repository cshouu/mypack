import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import FluidNet_2d4c_no_long
from torch.utils.tensorboard import SummaryWriter
import time
import json
import csv

# config
writer = SummaryWriter('runs')

F_TIME = time.strftime("%Y%m%d_%H%M%S", time.localtime())
LAST_MODEL_PATH = './'+F_TIME+'-last_presnet_state_dict.pth'
LAST_OPTIMIZER_PATH = './'+F_TIME+'-last_optimizer_state_dict.pth'
OPTIMAL_MODEL_PATH = './'+F_TIME+'-optimal_presnet_state_dict.pth'
TRAIN_CONF_PATH = './'+F_TIME+'-train_conf.json'
TRAIN_LOG_PATH = './'+F_TIME+'-train_log.csv'
with open(TRAIN_LOG_PATH, 'w') as logfile:
	w = csv.writer(logfile)
	w.writerow(['Epoch', 'Train_loss', 'Test_loss'])

train_conf = {
	'modelType': 'default',  # 'default' 'tog' 'yang'
	'normalizeInputChan': 0,  # 0:'pDiv' or 1:'UDiv' or 2:'div' normalizeInputFunc = 'std'
	'normalizeInputThreshold': 0.00001,
	'maxEpochs': 60,  # 5000
	'batch_size': 16,
	'lamda_ndiv': 0.0,
	'lamda_div': 0.6,
	'lamda_p': 0.2,
	'lamda_v': 0.2,
	'last_epoch': -1,
	'optimal_epoch': -1,
	'optimal_avg_train_loss': float('inf'),
	'optimal_avg_test_loss': float('inf')
}

# data
trainset = FluidNet_2d4c_no_long.VPGDdataset(isTrainset=True)
trainloader = DataLoader(trainset, batch_size=train_conf['batch_size'], shuffle=True, num_workers=2)
testset = FluidNet_2d4c_no_long.VPGDdataset(isTrainset=False)
testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

# model
presnet = FluidNet_2d4c_no_long.PressureNet(modelType=train_conf['modelType'])
optimizer = optim.Adam(presnet.parameters())
lossFunc = FluidNet_2d4c_no_long.LossFunc()

use_gpu = torch.cuda.is_available()
if use_gpu:
	presnet = presnet.cuda()
	lossFunc = lossFunc.cuda()

# init
presnet.load_state_dict(torch.load('./20200229_005124-presnet.pth'))
epoch = 1
time_zero = time.time()
if __name__ == '__main__':
	while epoch < train_conf['maxEpochs']:
		print('Epoch: %02d' % epoch)
		time_start = time.time()

		# Train code
		train_loss = 0.0
		for data_batched in trainloader:
			if use_gpu:
				data_batched['th_net_input'] = data_batched['th_net_input'].cuda()
				data_batched['th_vx'] = data_batched['th_vx'].cuda()
				data_batched['th_gx'] = data_batched['th_gx'].cuda()
				data_batched['th_p'] = data_batched['th_p'].cuda()
				data_batched['th_v'] = data_batched['th_v'].cuda()
				lamda_p = torch.tensor(train_conf['lamda_p']).cuda()
				lamda_v = torch.tensor(train_conf['lamda_v']).cuda()
				lamda_div = torch.tensor(train_conf['lamda_div']).cuda()
			optimizer.zero_grad()
			pres, vel = presnet(data_batched['th_net_input'], data_batched['th_vx'], data_batched['th_gx'], normalizeInputChan=train_conf['normalizeInputChan'], normalizeInputThrehsold=train_conf['normalizeInputThreshold'])
			loss = lossFunc(lamda_p=train_conf['lamda_p'], lamda_v=train_conf['lamda_v'], lamda_div=train_conf['lamda_div'], th_p=data_batched['th_p'], th_v=data_batched['th_v'], th_netp=pres, th_netv=vel)
			loss.backward()
			optimizer.step()
			if use_gpu:
				loss = loss.cpu()
			train_loss += loss.item()
		train_loss /= len(trainset)
		# test code
		test_loss = 0.0
		presnet.eval()
		with torch.no_grad():
			for data_batched in testloader:
				if use_gpu:
					data_batched['th_net_input'] = data_batched['th_net_input'].cuda()
					data_batched['th_vx'] = data_batched['th_vx'].cuda()
					data_batched['th_gx'] = data_batched['th_gx'].cuda()
					data_batched['th_p'] = data_batched['th_p'].cuda()
					data_batched['th_v'] = data_batched['th_v'].cuda()
				pres, vel = presnet(data_batched['th_net_input'], data_batched['th_vx'], data_batched['th_gx'], normalizeInputChan=train_conf['normalizeInputChan'], normalizeInputThrehsold=train_conf['normalizeInputThreshold'])
				loss = lossFunc(lamda_p=train_conf['lamda_p'], lamda_v=train_conf['lamda_v'], lamda_div=train_conf['lamda_div'], th_p=data_batched['th_p'], th_v=data_batched['th_v'], th_netp=pres, th_netv=vel)
				if use_gpu:
					loss = loss.cpu()
				test_loss += loss.item()
			test_loss /= len(testset)
		presnet.train()

		# save train and test log
		with open(TRAIN_LOG_PATH, 'a') as logfile:
			w = csv.writer(logfile)
			w.writerow([epoch, train_loss, test_loss])
		# write visiualization data
		writer.add_scalar('train loss', train_loss, epoch)
		writer.add_scalar('test loss', test_loss, epoch)
		for name, layer in presnet.named_parameters():
			writer.add_histogram(name + '_grad', layer.grad.data, epoch)
			writer.add_histogram(name + '_weight', layer.data, epoch)
		# save the last training model and update train conf
		if use_gpu:
			presnet = presnet.cpu()
		torch.save(presnet.state_dict(), LAST_MODEL_PATH)
		torch.save(optimizer.state_dict(), LAST_OPTIMIZER_PATH)
		if use_gpu:
			presnet = presnet.cuda()
		with open(TRAIN_CONF_PATH, 'w') as f:
			train_conf['last_epoch'] = epoch
			json.dump(train_conf, f)
		# save optimal model and update train conf
		if test_loss < train_conf['optimal_avg_test_loss']:
			if use_gpu:
				presnet = presnet.cpu()
			torch.save(presnet.state_dict(), OPTIMAL_MODEL_PATH)
			if use_gpu:
				presnet = presnet.cuda()
			with open(TRAIN_CONF_PATH, 'w') as f:
				train_conf['optimal_epoch'] = epoch
				train_conf['optimal_avg_test_loss'] = test_loss
				train_conf['optimal_avg_train_loss'] = train_loss
				json.dump(train_conf, f)

		time_end = time.time()
		print('  Train_loss: %.4f  Test_loss: %.4f  Delta_time: %dm  Total_time: %dm' % (train_loss, test_loss, (time_end-time_start)//60, (time_end-time_zero)//60))
		epoch += 1

