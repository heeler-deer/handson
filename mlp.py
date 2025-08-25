import torch
import torch.nn as nn
import torch.optim as optim

#定义一个MLP类，继承自torch.nn.Module
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2,output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.outlayer = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.outlayer(out)

# 定义模型参数
input_size = 768
hidden_size1 = 3072
hidden_size2 = 3072
output_size = 768

# 初始化MLP模型
MLP_model = MLP(input_size, hidden_size1, hidden_size2, output_size)

# 打印模型结构
print(MLP_model)

