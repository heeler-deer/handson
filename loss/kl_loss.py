import numpy as np

def kl_divergence_numpy(p, q):
    """
    计算两个离散概率分布 P 和 Q 之间的KL散度。

    参数:
    p (np.ndarray): 真实概率分布。必须为一维数组，元素非负且和为1。
    q (np.ndarray): 预测概率分布。必须为一维数组，元素非负且和为1。

    返回:
    float: KL散度的值。
    """
    # 为了数值稳定性，给 q 加上一个极小值 epsilon
    epsilon = 1e-10
    q = q + epsilon

    # 仅在 p(x) > 0 的地方计算，因为当 p(x) = 0 时，p(x)log(p(x)/q(x)) 的极限为 0
    # 这样可以避免 log(0) 的问题
    mask = p > 0
    p_masked = p[mask]
    q_masked = q[mask]

    # 根据公式计算 KL 散度
    return np.sum(p_masked * (np.log(p_masked) - np.log(q_masked)))

# --- 示例 ---
# 假设我们有两个概率分布
# P 是“真实”分布
p_dist = np.array([0.1, 0.2, 0.7])
# Q 是模型的“预测”分布
q_dist = np.array([0.2, 0.3, 0.5])

# 验证输入是合法的概率分布
assert np.isclose(np.sum(p_dist), 1.0), "P 分布的和不为1"
assert np.isclose(np.sum(q_dist), 1.0), "Q 分布的和不为1"


# 计算 KL 散度
kl_loss = kl_divergence_numpy(p_dist, q_dist)
print(f"NumPy 实现的 KL 散度 (P || Q): {kl_loss:.4f}")

# 另一个例子：当两个分布相同时，KL散度为0
p_same = np.array([0.1, 0.2, 0.7])
q_same = np.array([0.1, 0.2, 0.7])
kl_loss_same = kl_divergence_numpy(p_same, q_same)
print(f"当 P 和 Q 相同时的 KL 散度: {kl_loss_same:.4f}")

# 另一个例子：差异更大的分布
q_diff = np.array([0.8, 0.1, 0.1])
kl_loss_diff = kl_divergence_numpy(p_dist, q_diff)
print(f"差异更大时的 KL 散度 (P || Q_diff): {kl_loss_diff:.4f}")