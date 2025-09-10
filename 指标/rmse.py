import torch
import torch.nn.functional as F

def rmse_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    使用 PyTorch 计算 Root Mean Squared Error (RMSE).

    Args:
        y_pred (torch.Tensor): 模型预测值的张量.
        y_true (torch.Tensor): 真实值的张量.
    
    Returns:
        float: RMSE 的值.
    """
    # 确保张量具有相同的形状
    assert y_pred.shape == y_true.shape, "Input tensors must have the same shape"
    
    # 1. 使用 F.mse_loss 计算均方误差 (Mean Squared Error)
    #    F.mse_loss(input, target) = mean((input - target)^2)
    mse = F.mse_loss(y_pred, y_true)
    
    # 2. 对 MSE 开平方根得到 RMSE
    #    注意：需要确保 mse 不是负数，但 mse_loss 的定义保证了这一点
    rmse = torch.sqrt(mse)
    
    # 3. 使用 .item() 从单元素张量中提取 Python 数值
    return rmse.item()

# --- 示例 ---
# 模型的预测评分
y_pred_torch = torch.tensor([2.5, 4.0, 3.8])

# 用户的真实评分
y_true_torch = torch.tensor([3.0, 3.5, 5.0])

# 计算 RMSE
rmse_value = rmse_torch(y_pred_torch, y_true_torch)
print(f"PyTorch RMSE: {rmse_value}")
# 预期结果: sqrt(((3.0-2.5)^2 + (3.5-4.0)^2 + (5.0-3.8)^2) / 3)
#         = sqrt((0.25 + 0.25 + 1.44) / 3) = sqrt(1.94 / 3) ≈ 0.804
# 输出: PyTorch RMSE: 0.8041558265686035