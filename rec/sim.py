import numpy as np

# 假设 D=10, N=10000
user_vec = np.random.rand(10)
item_matrix = np.random.rand(10000, 10)
# 无需循环，一次性计算
# 1. 计算所有点积 (矩阵-向量乘法)
dot_products = item_matrix @ user_vec

# 2. 计算所有物品的模长 (沿特定轴计算)
item_norms = np.linalg.norm(item_matrix, axis=1)
user_norm = np.linalg.norm(user_vec)

# 3. 广播除法
similarities_vec = dot_products / (user_norm * item_norms)

print(similarities_vec)