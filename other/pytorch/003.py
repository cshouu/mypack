import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import torchvision.transforms as tfs

PATH = './cifar_net.pth'

transform = tfs.Compose([
	tfs.ToTensor(),
	tfs.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = tv.datasets.CIFAR10(root='./data',train=True,download=False,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset = tv.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform)
testloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1 = nn.Conv2d(3,6,5)
		self.conv2 = nn.Conv2d(6,16,5)
		self.pool = nn.MaxPool2d(2,2)
		self.fc1 = nn.Linear(16*5*5,120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)
	def forward(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1,16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

flag = int(input('input 0 for TRAIN or 1 for TEST:'))
if flag == 0:
	# train flow
	net = Net()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
	for epoch in range(2):
		running_loss = 0.0
		for i,data in enumerate(trainloader,0):
			inputs,labels = data
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs,labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if(i % 2000 == 1999):
				print('[%d,%5d] loss: %.3f' % (epoch+1,i+1,running_loss / 2000))
				running_loss = 0.0
	print('traing finished')
	torch.save(net.state_dict(),PATH)

elif flag == 1:
	# test flow
	net = Net()
	net.load_state_dict(torch.load(PATH))
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_,predicted = torch.max(outputs.data,1)
			c = (predicted == labels).squeeze()
			for i in range(4):
				label =labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1
	for i in range(10):
		print('Accuracy of %5s: %2d %%' % (classes[i],100*class_correct[i] / class_total[i]))

else:
	print('flag error')

