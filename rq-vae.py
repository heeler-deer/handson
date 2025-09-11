import torch, torch.nn as nn, torch.nn.functional as F

class RQ(nn.Module):
    """
    残差量化层，EMA 版，返回  z_q  +  indices（list of S 个 [N]）
    """
    def __init__(self, D, K, S, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.D, self.K, self.S = D, K, S
        self.beta = beta
        self.decay = decay
        self.eps = eps

        # 码本
        self.register_buffer('codebooks', torch.randn(S, K, D))
        # EMA 统计量
        self.register_buffer('N', torch.zeros(S, K))          # per-stage cluster count
        self.register_buffer('z_avg', torch.zeros(S, K, D))   # per-stage embed sum

    @torch.no_grad()
    def _update_ema(self, s, z_q, residual, indices):
        """EMA 更新第 s 级码本"""
        Ns, z_avgs = self.N[s], self.z_avg[s]          # [K] 和 [K, D]
        # one-hot 统计
        oh = F.one_hot(indices, self.K).float()        # [N, K]
        n_k = oh.sum(0)                                # [K]
        z_k = (oh.T @ residual) / (n_k + self.eps)     # [K, D]
        # 滑动平均
        Ns = self.decay * Ns + (1 - self.decay) * n_k
        z_avgs = self.decay * z_avgs + (1 - self.decay) * z_k
        # 写回
        self.N[s].copy_(Ns)
        self.z_avg[s].copy_(z_avgs)
        # 更新码本
        self.codebooks[s].copy_(z_avgs / (Ns + self.eps).unsqueeze(1))

    def forward(self, z):
        """
        z: [B, ..., D] ->  z_q 同形状
        return  z_q, loss, indices_list
        """
        shape = z.shape
        z = z.view(-1, self.D)          # [N, D]
        residual = z
        reconstructed = 0
        loss = 0.
        indices_list = []

        for s in range(self.S):
            C = self.codebooks[s]       # [K, D]
            # L2 距离
            dist = (residual.unsqueeze(1) - C.unsqueeze(0)).norm(dim=2)  # [N, K]
            idx = dist.argmin(1)                                       # [N]
            z_q = C[idx]                                               # [N, D]

            # 只有 commit loss 回传 encoder
            loss += F.mse_loss(z_q.detach(), residual) * self.beta

            indices_list.append(idx)
            reconstructed = reconstructed + z_q
            residual = residual - z_q.detach()

            # EMA 更新（训练模式）
            if self.training:
                self._update_ema(s, z_q, residual + z_q.detach(), idx)

        # straight-through
        z_q = z + (reconstructed - z).detach()
        z_q = z_q.view(*shape)
        return z_q, loss, indices_list
    
    
    
    
class RQVAE(nn.Module):
    def __init__(self, in_dim=784, hid=256, z_dim=64, K=512, S=4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, z_dim)
        )
        self.quant = RQ(z_dim, K, S)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hid), nn.ReLU(),
            nn.Linear(hid, in_dim), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.enc(x)
        z_q, q_loss, idx = self.quant(z)
        x_hat = self.dec(z_q)
        return x_hat, q_loss, idx
    
    
    
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.view(-1))])
train_set = datasets.MNIST(root='./mnist', train=True, transform=transform, download=True)
loader  = DataLoader(train_set, batch_size=256, shuffle=True)

model = RQVAE().to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

for epoch in range(5):
    total, rec_sum, q_sum = 0, 0., 0.
    for x, _ in loader:
        x = x.to(device)
        x_hat, q_loss, idx = model(x)
        rec_loss = F.mse_loss(x_hat, x)
        loss = rec_loss + q_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        total   += x.size(0)
        rec_sum += rec_loss.item() * x.size(0)
        q_sum   += q_loss.item() * x.size(0)

    # 码本利用率
    flat_idx = torch.cat(idx)
    usage = flat_idx.unique().numel()
    print(f'epoch {epoch+1}  rec={rec_sum/total:.4f}  q={q_sum/total:.4f}  usage={usage}/{model.quant.K*model.quant.S}')