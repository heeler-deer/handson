

import torch
import torch.nn.functional as F
import numpy as np
def infoNCE_loss(q, k, temperature=0.07):
    # L2 归一化
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)

    # 相似度矩阵
    logits = torch.matmul(q, k.t()) / temperature   # (N, N)

    # 正样本下标就是 [0,1,...,N-1]
    labels = torch.arange(q.size(0), device=q.device)

    # 交叉熵
    return F.cross_entropy(logits, labels)






