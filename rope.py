import torch
'''
https://blog.csdn.net/qq_30731313/article/details/146071289
'''
def rope(x):
    batch_size, seq_len, head_dim = x.shape
    device = x.device
    # 生成位置索引
    pos = torch.arange(seq_len, device=device).unsqueeze(1)
    # 计算 theta，生成不同维度的频率
    theta = 10000 ** (-torch.arange(0, head_dim, 2, device=device) / head_dim)

    # 计算旋转角度
    angles = pos * theta

    # 计算 cos 和 sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # 拆分输入数据，每两个维度为一组
    x1, x2 = x[..., 0::2], x[..., 1::2]
    
    # 应用旋转变换
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # 拼接回去，形成完整的旋转后向量
    return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

# 测试代码
batch_size, seq_len, head_dim = 2, 5, 4
x = torch.randn(batch_size, seq_len, head_dim)
x_rope = rope(x)

print("原始x:", x)
print("旋转后x:", x_rope)
