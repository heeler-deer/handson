import torch
import torch.nn.functional as F

class RQKMeans(torch.nn.Module):
    def __init__(self, dim, num_levels=3, codebook_size=256, kmeans_iters=10):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        # 初始化 codebooks，形状 (num_levels, codebook_size, dim)
        self.codebooks = torch.nn.Parameter(torch.randn(num_levels, codebook_size, dim))

    def forward(self, x):
        """
        输入:
            x：形状 (B, dim)
        输出:
            recon: (B, dim) 重构后的向量
            codes: (B, num_levels) 每层簇索引
        """
        B, D = x.shape
        residual = x
        codes = []
        recon = 0

        for l in range(self.num_levels):
            # 计算与当前 codebook 的距离
            cb = self.codebooks[l]  # (K, D)
            # 使用欧氏距离： (B, K)
            dist = torch.cdist(residual, cb)
            idx = torch.argmin(dist, dim=1)  # (B,)
            codes.append(idx)

            # 获取对应中心
            selected = cb[idx]  # (B, D)
            recon = recon + selected
            residual = residual - selected  # 更新 residual

        codes = torch.stack(codes, dim=1)  # (B, num_levels)
        return recon, codes

    def kmeans_init(self, x):
        """
        可选：使用简单的 K-Means 初始化 codebook（每层）
        """
        from sklearn.cluster import KMeans
        x_np = x.detach().cpu().numpy()
        for l in range(self.num_levels):
            kmeans = KMeans(n_clusters=self.codebook_size, n_init=1, max_iter=self.kmeans_iters)
            kmeans.fit(x_np)
            centroids = torch.Tensor(kmeans.cluster_centers_)  # (K, D)
            with torch.no_grad():
                self.codebooks[l].copy_(centroids)
            # 更新 residual
            labels = kmeans.predict(x_np)
            x_np = x_np - centroids[labels]

# 使用示例
if __name__ == "__main__":
    B, D = 128, 64
    x = torch.randn(B, D)
    model = RQKMeans(dim=D, num_levels=3, codebook_size=64)
    # 可选初始化
    model.kmeans_init(x)

    recon, codes = model(x)
    loss = F.mse_loss(recon, x)
    loss.backward()
    print("重构误差:", loss.item())
    print("编码索引形状:", codes.shape)
