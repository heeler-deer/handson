import numpy as np
import torch
import torch.nn.functional as F



def margin_loss(x1, x2, y, margin=1.0):
    # 欧氏距离
    dist = np.linalg.norm(x1 - x2, axis=1)
    
    # 正负样本 loss
    loss = y * dist**2 + (1 - y) * np.maximum(0, margin - dist)**2
    return np.mean(loss)



def margin_loss_torch(x1, x2, y, margin=1.0):
    # 计算欧氏距离
    dist = F.pairwise_distance(x1, x2, p=2)
    
    # 对正负样本分别计算损失
    loss = y * dist**2 + (1 - y) * F.relu(margin - dist)**2
    return loss.mean()
