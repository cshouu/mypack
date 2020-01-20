import torch
import torch.nn

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1 = nn.Conv2D(3,10,3)

	def forward(self,x):
		out = self.conv1(x)
		return out
