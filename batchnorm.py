import torch
from torch import nn
import numpy as np



class BN(nn.Module):
	def __init__(self, num_features, momentum=0.01, eps=1e-5):
		super().__init__()
		self.running_mean = torch.zeros(num_features)
		self.running_var = torch.ones(num_features)
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum
		self.gamma = nn.Parameter(torch.ones(num_features))  
		self.beta = nn.Parameter(torch.zeros(num_features))

	def forward(self, x):
		if self.training:
			#[B,D]
			x_mean = x.mean(dim=0, keepdim=True)
			x_var = x.var(dim=0, keepdim=True, unbiased=False)
			self.running_mean = (1 - self.momentum) * x_mean + self.momentum * self.running_mean
			self.running_var = (1 - self.momentum) * x_var + self.momentum * self.running_var
			x_normalized = (x - x_mean) / np.sqrt(x_var+self.eps)
		else:
			x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
		return self.gamma * x_normalized + self.beta