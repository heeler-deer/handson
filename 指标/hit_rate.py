import torch

def hit_ratio_at_k_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    使用 PyTorch 计算 Hit Ratio @ K.

    Args:
        y_pred (torch.Tensor): 推荐结果的二维张量, shape (n_users, k).
        y_true (torch.Tensor): 真实结果的一维张量, shape (n_users,).
    
    Returns:
        float: Hit Ratio @ K 的值.
    """
    # 使用 unsqueeze(1) 将 y_true 变形为 (n_users, 1) 以便广播
    y_true = y_true.unsqueeze(1)
    
    # 比较并检查每一行是否有命中
    hits = torch.any(y_pred == y_true, dim=1)
    
    # 计算命中比例, .item() 用于从单元素张量中提取 Python 数值
    return hits.float().mean().item()

# --- 示例 ---
# 推荐的物品ID列表
y_pred_torch = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
    [11, 22, 33],
    [44, 55, 66]
])

# 真实的用户交互物品ID
y_true_torch = torch.tensor([20, 15, 90, 44, 55])

hr_torch = hit_ratio_at_k_torch(y_pred_torch, y_true_torch)
print(f"PyTorch Hit Ratio @ 3: {hr_torch}")
# 输出: PyTorch Hit Ratio @ 3: 0.6000000238418579 (由于浮点数精度，可能略有差异)