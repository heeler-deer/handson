import torch
from torch import nn
import numpy as np

class LN(nn.Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - x_mean) / torch.sqrt(x_var + self.eps)
        return self.gamma * x_normalized + self.beta