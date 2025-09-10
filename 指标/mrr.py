import torch

def mrr_at_k_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    使用 PyTorch 计算 Mean Reciprocal Rank @ K.

    Args:
        y_pred (torch.Tensor): 推荐结果的二维张量, shape (n_users, k).
                             每一行是一个用户的 top-k 推荐物品ID列表。
        y_true (torch.Tensor): 真实结果的一维张量, shape (n_users,).
                             每个元素是对应用户的真实交互物品ID。
    
    Returns:
        float: MRR @ K 的值.
    """
    # 使用 unsqueeze(1) 将 y_true 变形为 (n_users, 1) 以便广播
    y_true_reshaped = y_true.unsqueeze(1)
    
    # 比较 y_pred 和 y_true，得到一个布尔张量，标记命中的位置
    # shape: (n_users, k)
    hits = (y_pred == y_true_reshaped)
    
    # 找到每个用户第一次命中的位置（0-based index）
    # argmax 在找到第一个 True 后就会停止。如果一行全是 False，它会返回 0。
    # shape: (n_users,)
    hit_indices = torch.argmax(hits.int(), dim=1)
    
    # 检查哪些用户是真正命中了的（避免将全False的0误认为命中）
    # shape: (n_users,)
    has_hits = torch.any(hits, dim=1)
    
    # 计算排名（1-based rank），排名 = 索引 + 1
    ranks = (hit_indices + 1).float()
    
    # 计算倒数排名，对于没有命中的用户，其倒数排名应为0
    # 我们通过将 has_hits (布尔型) 转为浮点型（True->1.0, False->0.0）来实现
    reciprocal_ranks = (1.0 / ranks) * has_hits.float()
    
    # 计算所有用户倒数排名的平均值
    return reciprocal_ranks.mean().item()

# --- 示例 ---
# 推荐的物品ID列表 (5个用户, K=4)
y_pred_torch = torch.tensor([
    [10, 20, 30, 40], # 用户0: 命中在第2位 (rank=2) -> RR = 0.5
    [50, 60, 70, 80], # 用户1: 未命中 -> RR = 0
    [90, 11, 22, 33], # 用户2: 命中在第1位 (rank=1) -> RR = 1.0
    [44, 55, 66, 77], # 用户3: 命中在第4位 (rank=4) -> RR = 0.25
    [88, 99, 10, 20]  # 用户4: 未命中 -> RR = 0
])

# 真实的用户交互物品ID
y_true_torch = torch.tensor([20, 15, 90, 77, 100])

# 预期 MRR = (0.5 + 0 + 1.0 + 0.25 + 0) / 5 = 1.75 / 5 = 0.35
mrr = mrr_at_k_torch(y_pred_torch, y_true_torch)
print(f"PyTorch MRR @ 4: {mrr}")
# 输出: PyTorch MRR @ 4: 0.3499999940395355