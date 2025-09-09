import torch





import torch

def l1_norm(x, eps=1e-6):
    x = x / torch.clamp(
        torch.linalg.norm(x, ord=1, dim=-1, keepdim=True),
        min=eps,
    )
    return x

def l2_norm(x, eps=1e-6):
    x = x / torch.clamp(
        torch.linalg.norm(x, ord=2, dim=-1, keepdim=True),
        min=eps,
    )
    return x