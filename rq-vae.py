import torch
import torch.nn as nn
import torch.nn.functional as F

class RQ_k_means(nn.Module):
    """
    Residual Quantization (multi-stage) layer — 基于 VQ-VAE 的常见 loss 设计。
    Inputs:
        codebook_size: K
        codebook_dim: D
        num_stages: number of residual stages
        commit_weight: beta for commitment loss
    Returns:
        quantized_z: tensor with straight-through estimator (same shape as input z)
        quantization_loss: embedding_loss + beta * commitment_loss
    """
    def __init__(self, codebook_size, codebook_dim, num_stages=4, commit_weight=0.25):
        super().__init__()
        self.K = codebook_size
        self.D = codebook_dim
        self.S = num_stages
        self.commit_weight = commit_weight

        # 用一个参数张量保存所有 stage 的码本： shape [S, K, D]
        # 也可以用 ParameterList，但一个大张量便于实现 EMA 时的统计
        self.codebooks = nn.Parameter(torch.randn(self.S, self.K, self.D) * 0.1)

    def forward(self, z):
        """
        z: [B, ..., D] 或 [B, D]
        返回 quantized_z 的形状与 z 相同
        """
        orig_shape = z.shape
        # 将 z 展平为 [N, D]，其中 N = B * spatial
        z_flat = z.view(-1, self.D)  # [N, D]
        N = z_flat.size(0)

        residual = z_flat  # 初始残差
        reconstructed = torch.zeros_like(z_flat)
        embedding_loss = 0.0
        commitment_loss = 0.0

        for s in range(self.S):
            codebook = self.codebooks[s]  # [K, D]

            # 计算 L2 距离： [N, K]
            # 更稳健、直观的写法
            # residual.unsqueeze(1) -> [N,1,D]; codebook.unsqueeze(0) -> [1,K,D]
            dists = torch.sum((residual.unsqueeze(1) - codebook.unsqueeze(0)) ** 2, dim=-1)  # [N, K]

            # 取最近的码本索引
            indices = torch.argmin(dists, dim=1)  # [N]

            # 根据索引取出量化向量（会跟随 codebook 的参数）
            quantized = codebook[indices]  # [N, D]

            # 计算 VQ 两项 loss（注意 detach 的使用）
            # embedding_loss：把码本向量拉向编码器输出（梯度流向码本）
            embedding_loss = embedding_loss + F.mse_loss(quantized, residual.detach())

            # commitment_loss：把编码器的输出拉向选中码本（梯度流向编码器）
            commitment_loss = commitment_loss + F.mse_loss(quantized.detach(), residual)

            # 更新重构与残差（注意用 detach() 防止影响码本更新路径）
            reconstructed = reconstructed + quantized
            residual = residual - quantized.detach()

        # 总的量化损失
        quantization_loss = embedding_loss + self.commit_weight * commitment_loss

        # straight-through estimator：让梯度直接从 quantized 流回 encoder 的 z_flat
        # 公式与 VQ-VAE 一致：z_q = z + (quantized - z).detach()
        # 这里用 reconstructed 作为 quantized（多阶段求和的结果）
        z_q_flat = z_flat + (reconstructed - z_flat).detach()

        # 恢复原始形状
        z_q = z_q_flat.view(*orig_shape)

        return z_q, quantization_loss

# --- Step 2: 构建 RQ-VAE 模型 ---
class RQ_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_stages, codebook_size):
        super(RQ_VAE, self).__init__()

        # 编码器：将输入映射到潜在空间
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 残差量化层
        self.quantizer = RQ_k_means(codebook_size, latent_dim, num_stages)
        
        # 解码器：将量化后的潜在向量映射回输入空间
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() # 适用于[0,1]范围的数据，例如图像
        )

    def forward(self, x):
        # 编码
        z_continuous = self.encoder(x)
        
        # 残差量化
        z_quantized, quantization_loss = self.quantizer(z_continuous)
        
        # 解码
        x_recon = self.decoder(z_quantized)
        
        return x_recon, z_continuous, quantization_loss

# --- Step 3: 训练流程 ---
def train_rq_vae(model, dataloader, epochs, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for i, (data, _) in enumerate(dataloader):
            data = data.view(data.size(0), -1)  # 展平数据

            x_recon, z_continuous, q_loss = model(data)

            # 重构损失：测量解码器输出与原始输入之间的差异
            reconstruction_loss = F.mse_loss(x_recon, data)

            # VAE损失：重构损失 + KL散度（KL divergence）
            # 注意：这里我们省略了KL散度，因为它在离散潜在空间中不适用
            # 但是，VQ-VAE论文使用了一个 "commit loss" 来替代，
            # 我们在 RQ_k_means 层中已经包含了类似的概念（q_loss）
            loss = reconstruction_loss + q_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {loss.item():.4f}, Recon Loss: {reconstruction_loss.item():.4f}, Q Loss: {q_loss.item():.4f}")

# --- 运行示例（需要安装torchvision和下载MNIST数据集） ---
if __name__ == '__main__':
    try:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader

        # 超参数
        INPUT_DIM = 28 * 28  # MNIST图像大小
        HIDDEN_DIM = 256
        LATENT_DIM = 128
        NUM_STAGES = 4
        CODEBOOK_SIZE = 512

        # 数据集
        dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        # 初始化模型
        model = RQ_VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, NUM_STAGES, CODEBOOK_SIZE)

        # 训练
        train_rq_vae(model, dataloader, epochs=5)
        
    except ImportError:
        print("请安装 torchvision 以运行此示例: pip install torchvision")