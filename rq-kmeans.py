import numpy as np

def l2(x, y):
    """x: [N,D]  y: [K,D]  ->  [N,K]"""
    return np.sum((x[:, None, :] - y[None, :, :])**2, axis=2)

def kmeans_once(X, K, n_iter=20):
    """单次 KMeans，返回质心  [K,D] 与索引  [N]"""
    N, D = X.shape
    C = X[np.random.choice(N, K, replace=False)]   # 随机初始化
    for _ in range(n_iter):
        dist = l2(X, C)          # [N,K]
        idx = dist.argmin(1)     # [N]
        # 重新计算质心
        for k in range(K):
            mask = idx == k
            if mask.sum():
                C[k] = X[mask].mean(0)
            else:                # 空簇重随机
                C[k] = X[np.random.choice(N)]
    return C, idx


def rq_kmeans(X, K, S=4, n_iter=10):
    """
    X: [N,D]  原始数据
    K: 每级码本大小
    S: 残差级数
    return
        codebooks: [S,K,D]  每级质心
        indices:   [N,S]     每级最近邻索引
        X_rec:     [N,D]     重建结果
    """
    N, D = X.shape
    codebooks = np.zeros((S, K, D))
    indices = np.zeros((N, S), dtype=int)

    residual = X.copy()
    for s in range(S):
        # 对当前残差做 KMeans
        C, idx = kmeans_once(residual, K, n_iter)
        codebooks[s] = C
        indices[:, s] = idx

        # 重建并更新残差
        quantized = C[idx]          # [N,D]
        residual = residual - quantized

    # 最终重建 = 所有量化向量之和
    X_rec = codebooks[range(S), indices].sum(1)   # [N,D]
    return codebooks, indices, X_rec


# 随机数据
X = np.random.randn(1000, 64).astype(np.float32)

codebooks, indices, X_rec = rq_kmeans(X, K=128, S=4)

# 计算重建误差
mse = ((X - X_rec)**2).mean()
print("RQ-KMeans MSE:", mse)

