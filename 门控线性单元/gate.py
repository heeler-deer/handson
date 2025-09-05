import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# 1. 原始 GLU (Gated Linear Unit) 实现
# 使用 Sigmoid 作为门控激活函数
# ----------------------------------------
class OriginalGLU(nn.Module):
    """
    原始的门控线性单元。
    公式: GLU(x, W, V) = σ(xW) ⊗ (xV)
    """
    def __init__(self, dim_in: int, dim_out: int):
        """
        Args:
            dim_in (int): 输入特征的维度。
            dim_out (int): 输出特征的维度。
        """
        super().__init__()
        # 单个线性层，将输入投影到输出维度的两倍
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., dim_in)
        
        # 1. 线性投影
        x_proj = self.proj(x)  # shape: (..., dim_out * 2)
        
        # 2. 分割成 gate 和 value 两部分
        gate, value = x_proj.chunk(2, dim=-1)  # 两个 tensor 的 shape 都是 (..., dim_out)
        
        # 3. 对 gate 应用 Sigmoid 激活函数并与 value 相乘
        return F.sigmoid(gate) * value

# ----------------------------------------
# 2. GeGLU (Gaussian-Error Linear Unit GLU) 实现
# 使用 GELU 作为门控激活函数
# ----------------------------------------
class GeGLU(nn.Module):
    """
    使用 GELU 激活的门控线性单元。
    公式: GeGLU(x, W, V) = GELU(xW) ⊗ (xV)
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.proj(x)
        gate, value = x_proj.chunk(2, dim=-1)
        # 使用 GELU 激活函数
        return F.gelu(gate) * value

# ----------------------------------------
# 3. SwiGLU (Swish-Gated Linear Unit) 实现
# 使用 SiLU (Swish) 作为门控激活函数
# 这是目前大语言模型中最常用、效果最好的变体
# ----------------------------------------
class SwiGLU(nn.Module):
    """
    使用 SiLU (Swish) 激活的门控线性单元。
    公式: SwiGLU(x, W, V) = SiLU(xW) ⊗ (xV)
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.proj(x)
        gate, value = x_proj.chunk(2, dim=-1)
        # F.silu 是 SiLU/Swish 的函数式 API
        return F.silu(gate) * value

# ----------------------------------------
# 使用示例
# ----------------------------------------
if __name__ == '__main__':
    # 定义模型参数
    batch_size = 4
    seq_len = 10
    input_dim = 128  # 输入维度
    output_dim = 256 # 输出维度

    # 创建一个随机输入张量
    input_tensor = torch.randn(batch_size, seq_len, input_dim)
    print(f"输入张量尺寸: {input_tensor.shape}\n")

    # 1. 测试 OriginalGLU
    glu_layer = OriginalGLU(dim_in=input_dim, dim_out=output_dim)
    glu_output = glu_layer(input_tensor)
    print("--- OriginalGLU ---")
    print(f"输出张量尺寸: {glu_output.shape}")
    assert glu_output.shape == (batch_size, seq_len, output_dim)
    print("尺寸正确!\n")

    # 2. 测试 GeGLU
    geglu_layer = GeGLU(dim_in=input_dim, dim_out=output_dim)
    geglu_output = geglu_layer(input_tensor)
    print("--- GeGLU ---")
    print(f"输出张量尺寸: {geglu_output.shape}")
    assert geglu_output.shape == (batch_size, seq_len, output_dim)
    print("尺寸正确!\n")

    # 3. 测试 SwiGLU
    swiglu_layer = SwiGLU(dim_in=input_dim, dim_out=output_dim)
    swiglu_output = swiglu_layer(input_tensor)
    print("--- SwiGLU ---")
    print(f"输出张量尺寸: {swiglu_output.shape}")
    assert swiglu_output.shape == (batch_size, seq_len, output_dim)
    print("尺寸正确!\n")