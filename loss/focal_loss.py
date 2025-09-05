import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassFocalLoss(nn.Module):
    """
    Multi-class Focal Loss (softmax + CE 风格)
    Args:
        alpha (float or list, optional): 类别平衡因子，float 表示对所有类相同，
                                         list/ndarray 表示对每个类别单独设置。
        gamma (float): 调节难易样本的指数因子，默认 2.0
        reduction (str): {'none', 'mean', 'sum'}，默认 'mean'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        if isinstance(alpha, (list, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, C] logits (未 softmax)
            targets: [N] 类别索引 (0 ~ C-1)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")  # [N]
        log_probs = F.log_softmax(inputs, dim=1)                      # [N, C]
        probs = torch.exp(log_probs)                                  # [N, C]

        # 取出每个样本对应类别的概率
        targets_onehot = F.one_hot(targets, num_classes=inputs.size(1)).float()  # [N, C]
        pt = (probs * targets_onehot).sum(dim=1)  # [N]

        # focal loss 调节项
        focal_factor = (1 - pt) ** self.gamma
        loss = focal_factor * ce_loss

        # alpha 平衡因子
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                at = self.alpha[targets]  # 按类别取 alpha
            else:
                at = self.alpha
            loss = at * loss

        # reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


criterion = MultiClassFocalLoss(alpha=[1, 2, 1, 1, 1], gamma=2.0, reduction="mean")

# 假设 batch_size = 3, num_classes = 5
inputs = torch.randn(3, 5, requires_grad=True)  # logits
targets = torch.tensor([1, 2, 0])               # 类别索引

loss = criterion(inputs, targets)
print(loss.item())
