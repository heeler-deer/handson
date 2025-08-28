##
## https://blog.csdn.net/weixin_43427721/article/details/134539003
##


import torch
import torch.nn.functional as F
import numpy as np
def approx_infoNCE_loss(q, k):
    # 计算query和key的相似度得分
    similarity_scores = torch.matmul(q, k.t())  # 矩阵乘法计算相似度得分
    # 计算相似度得分的温度参数
    temperature = 0.07
    # 计算logits
    logits = similarity_scores / temperature
    # 构建labels（假设有N个样本）
    N = q.size(0)
    labels = torch.arange(N).to(logits.device)
    # 计算交叉熵损失
    loss = F.cross_entropy(logits, labels)
    return loss





if __name__=='__main__':
    # 生成一些随机数据用于测试
    N = 4  # 批次大小 (Batch Size)
    embedding_dim = 128  # 向量维度

    # 随机生成 query 和 key
    query_vectors = np.random.rand(N, embedding_dim)
    key_vectors = np.random.rand(N, embedding_dim)

    # 标准化向量 (在对比学习中，通常会对向量进行L2归一化)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    key_vectors = key_vectors / np.linalg.norm(key_vectors, axis=1, keepdims=True)
    q_torch = torch.tensor(query_vectors)
    k_torch = torch.tensor(key_vectors)

    similarity_scores = torch.matmul(q_torch, k_torch.t())
    temperature = 0.07
    logits = similarity_scores / temperature
    labels = torch.arange(N)
    torch_loss = F.cross_entropy(logits, labels)

    print(f"PyTorch InfoNCE Loss: {torch_loss.item()}")
