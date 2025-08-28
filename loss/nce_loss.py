import torch
import torch.nn.functional as F

def nce_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    positive_noise_log_probs: torch.Tensor,
    negative_noise_log_probs: torch.Tensor,
    num_negative_samples: int
) -> torch.Tensor:

    # 将正负样本的 logits 和噪声概率拼接在一起
    all_logits = torch.cat([positive_logits, negative_logits], dim=1)
    all_noise_log_probs = torch.cat([positive_noise_log_probs, negative_noise_log_probs], dim=1)
    # log(k * Pn(w)) = log(k) + log(Pn(w))
    k_log_prob = torch.log(torch.tensor(num_negative_samples, dtype=torch.float, device=all_logits.device))
    classifier_logits = all_logits - (k_log_prob + all_noise_log_probs)

    # 创建二分类的标签
    # 正样本标签为 1，负样本标签为 0
    batch_size = positive_logits.shape[0]
    labels = torch.zeros_like(classifier_logits)
    labels[:, 0] = 1.0  # 第一列是正样本

    # 使用带有 logits 的二元交叉熵损失，以获得更好的数值稳定性
    # F.binary_cross_entropy_with_logits combines a Sigmoid layer and the BCELoss in one single class.
    loss = F.binary_cross_entropy_with_logits(classifier_logits, labels)
    
    return loss




batch_size = 8
embedding_dim = 128
num_negative_samples = 5 # k=5

positive_logits = torch.randn(batch_size, 1) 

negative_logits = torch.randn(batch_size, num_negative_samples)

# --- 模拟噪声分布概率 ---
# 这里我们用随机数模拟 log(P_n(w))
positive_noise_log_probs = torch.log(torch.rand(batch_size, 1))
negative_noise_log_probs = torch.log(torch.rand(batch_size, num_negative_samples))

# --- 计算 NCE Loss ---
loss_value = nce_loss(
    positive_logits,
    negative_logits,
    positive_noise_log_probs,
    negative_noise_log_probs,
    num_negative_samples
)

print(f"NCE Loss: {loss_value.item()}")