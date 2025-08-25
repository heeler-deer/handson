
import numpy as np
import random
def GOSS(data, gradients, a, b):
    # 样本数量
    n = len(data)
    
    # 计算梯度的绝对值
    abs_gradients = [abs(g) for g in gradients]
    
    # 按梯度的绝对值进行排序
    sorted_indices = sorted(range(n), key=lambda i: abs_gradients[i], reverse=True)
    
    # 保留大梯度样本
    top_k = int(a * n)
    large_gradient_indices = sorted_indices[:top_k]
    
    # 对小梯度样本进行随机采样
    remaining_indices = sorted_indices[top_k:]
    small_gradient_sample_size = int(b * n)
    small_gradient_indices = random.sample(remaining_indices, small_gradient_sample_size)
    
    # 合并大梯度样本和小梯度样本
    sampled_indices = large_gradient_indices + small_gradient_indices
    
    # 根据采样的索引获取采样后的数据集
    sampled_data = [data[i] for i in sampled_indices]
    
    # 对小梯度样本进行权重调整（使其对模型的影响等效于其在原始数据中的影响）
    scale_factor = (1 - a) / b
    for i in small_gradient_indices:
        sampled_data[i].weight *= scale_factor
    
    return sampled_data