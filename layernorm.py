import torch
from torch import nn
import numpy as np

class LN(nn.Module):
	def __init__(self, features, eps=1e-5):
		super().__init__()
		self.eps = eps
		# 对应LN中需要更新的beta和gamma，采用pytorch文档中的初始化值
		self.gamma = nn.Parameter(torch.ones(features))  # 缩放参数
		self.beta = nn.Parameter(torch.zeros(features))  # 偏移参数
	def forward(self, x):
		x_mean = x.mean(dim=-1, keepdim=True)
		x_var = x.var(dim=-1, keepdim=True, unbiased=False)
		return self.gamma * (x - x_mean) / np.sqrt(x_var+self.eps) + self.beta