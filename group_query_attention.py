##
## https://yuanchaofa.com/hands-on-code/hands-on-group-query-attention-and-multi-query-attention.html#multi-head-self-attention
##

import torch
import torch.nn as nn
import math

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, nums_key_value_head, dropout_p=0.0):
        """
        初始化 Grouped-Query Attention 模块.

        参数:
            hidden_dim (int): 输入的隐藏层维度.
            nums_head (int): 查询头的数量 (Q).
            nums_key_value_head (int): 键/值头的数量 (K, V).
            dropout_p (float): attention weight 的 dropout 概率.
        """
        super().__init__()
        # 确保维度可以被整除
        assert hidden_dim % nums_head == 0, "hidden_dim must be divisible by nums_head"
        # 确保 Q-head 的数量是 KV-head 的整数倍
        assert nums_head % nums_key_value_head == 0, "nums_head must be divisible by nums_key_value_head"

        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_key_value_head = nums_key_value_head
        self.head_dim = hidden_dim // nums_head

        # 初始化 q, k, v, o 的线性投射层
        self.q_proj = nn.Linear(hidden_dim, nums_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(nums_head * self.head_dim, hidden_dim, bias=False)
        
        # 初始化 Attention Dropout
        self.attention_dropout = nn.Dropout(dropout_p)

    def forward(self, X, attention_mask=None):
        """
        前向传播.

        参数:
            X (torch.Tensor): 输入张量, shape (batch_size, seq_len, hidden_dim).
            attention_mask (torch.Tensor, optional): 注意力掩码, 用于忽略 padding token.
                                                     shape (batch_size, 1, 1, seq_len)
                                                     或 (batch_size, seq_len, seq_len).
                                                     掩码中值为 0 或 False 的位置将被忽略.
        返回:
            torch.Tensor: attention 层的输出, shape (batch_size, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = X.size()

        # 1. 线性投射 Q, K, V
        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)

        # 2. 调整 Q, K, V 的形状以适应多头注意力
        # q: (batch_size, seq_len, nums_head, head_dim) -> (batch_size, nums_head, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        # k: (batch_size, seq_len, nums_key_value_head, head_dim) -> (batch_size, nums_key_value_head, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.nums_key_value_head, self.head_dim).transpose(1, 2)
        # v: (batch_size, seq_len, nums_key_value_head, head_dim) -> (batch_size, nums_key_value_head, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.nums_key_value_head, self.head_dim).transpose(1, 2)

        # 3. GQA 核心: 重复 K 和 V 头以匹配 Q 头的数量
        # repeat_interleave 会沿着指定维度(dim=1)重复张量
        # 每个 KV 头会服务于 num_groups 个 Q 头
        num_groups = self.nums_head // self.nums_key_value_head
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)

        # 4. 计算注意力分数
        # (b, n_h, seq, h_d) @ (b, n_h, h_d, seq) -> (b, n_h, seq, seq)
        attention_score = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 5. [新增] 应用 attention_mask (在 softmax 之前)
        if attention_mask is not None:
            # 将 mask 中为 0 (padding) 的位置设置为一个极小的负数
            # 这样在 softmax 后，这些位置的概率会趋近于 0
            attention_score = attention_score.masked_fill(attention_mask == 0, -1e9)

        # 6. 计算注意力权重 (softmax)
        attention_weight = torch.softmax(attention_score, dim=-1)
        
        # 7. [新增] 应用 attention_dropout (在 softmax 之后)
        attention_weight = self.attention_dropout(attention_weight)

        # 8. 使用注意力权重加权 V
        # (b, n_h, seq, seq) @ (b, n_h, seq, h_d) -> (b, n_h, seq, h_d)
        output = attention_weight @ v

        # 9. 重新组合多头输出并进行最终的线性投射
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        final_output = self.o_proj(output)

        return final_output

# ============= 测试代码 =============
if __name__ == '__main__':
    batch_size = 4
    seq_len = 10
    hidden_dim = 128
    nums_head = 8
    nums_kv_head = 4

    # 创建一个随机输入
    x = torch.rand(batch_size, seq_len, hidden_dim)

    # 创建一个 padding mask 示例
    # 假设每个样本的有效长度不同
    # 1 表示有效 token, 0 表示 padding token
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[1, 7:] = 0  # 第2个样本从第7个 token 开始是 padding
    attention_mask[2, 5:] = 0  # 第3个样本从第5个 token 开始是 padding
    attention_mask[3, 8:] = 0  # 第4个样本从第8个 token 开始是 padding
    
    # 将 mask 调整为 attention score 的形状 (b, 1, 1, seq) 以便广播
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    print("Mask shape:", attention_mask.shape)
    
    # 初始化模型，加入 dropout
    net = GroupQueryAttention(hidden_dim, nums_head, nums_kv_head, dropout_p=0.1)
    
    # 设置为评估模式以禁用 dropout (如果需要)
    # net.eval() 
    
    # 前向传播
    output = net(x, attention_mask=attention_mask)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    assert output.shape == x.shape # 确保输出形状与输入一致