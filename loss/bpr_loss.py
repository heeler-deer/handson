import torch
import torch.nn.functional as F

def bpr_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    """
    计算 BPR (Bayesian Personalized Ranking) Loss。
    该函数旨在让正样本的得分高于负样本的得分。
    参数:
    positive_scores (torch.Tensor): 一个批次中正样本的得分，形状为 (batch_size, 1) 或 (batch_size,).
    negative_scores (torch.Tensor): 一个批次中负样本的得分，形状为 (batch_size, 1) 或 (batch_size,).
    返回:
    torch.Tensor: 计算出的 BPR loss，一个标量 (scalar)。
    """
    # 1. 计算正样本得分与负样本得分的差值
    # 我们希望这个差值尽可能大
    score_diff = positive_scores - negative_scores

    # 2. 计算 BPR loss
    # L = -log(sigmoid(score_diff))
    # 为了数值稳定性，我们使用 F.softplus(-score_diff)
    # 这在数学上等价于 -log(sigmoid(score_diff))
    loss = F.softplus(-score_diff)

    # 3. 返回批次中所有样本对的平均损失
    return loss.mean()

# --- 备选实现（使用 log_sigmoid）---
def bpr_loss_alternative(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    """BPR Loss 的另一种实现方式，使用 log_sigmoid。"""
    score_diff = positive_scores - negative_scores
    # F.logsigmoid(x) 计算 log(sigmoid(x))
    # 因此我们需要取负数来得到 -log(sigmoid(x))
    loss = -F.logsigmoid(score_diff)
    return loss.mean()