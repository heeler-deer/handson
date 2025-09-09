import numpy as np

# 1. ReLU
def relu(x):
    return np.maximum(0, x)

# 2. Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 3. Tanh
def tanh(x):
    return np.tanh(x)

# 4. Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# 5. ELU
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# 6. GELU (近似公式)
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

# 7. Swish (SiLU)
def swish(x):
    return x * sigmoid(x)

# 8. Softplus
def softplus(x):
    return np.log1p(np.exp(x))  # log(1 + e^x)

# 9. Softmax
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # 防止溢出
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# 10. Mish
def mish(x):
    return x * np.tanh(softplus(x))

# 11. dice
def dice_activation(x, axis=-1, epsilon=1e-8):
    # 1. 沿指定轴计算均值和方差
    # keepdims=True 确保均值和方差的形状能与 x 进行广播
    mean = np.mean(x, axis=axis, keepdims=True)
    variance = np.var(x, axis=axis, keepdims=True)
    
    # 2. 计算动态斜率 alpha (在公式中称为 p(s))
    # 这是一个与 x 形状相同的数组
    alpha = 1 / (1 + np.exp(-(x - mean) / np.sqrt(variance + epsilon)))
    
    # 3. 应用 Dice 激活函数
    # 条件：x > 0
    # 如果为 True，则值为 x
    # 如果为 False，则值为 alpha * x
    output = np.where(x > 0, x, alpha * x)
    
    return output


x = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])

print("ReLU:", relu(x))
print("Sigmoid:", sigmoid(x))
print("Tanh:", tanh(x))
print("Leaky ReLU:", leaky_relu(x))
print("ELU:", elu(x))
print("GELU:", gelu(x))
print("Swish:", swish(x))
print("Softplus:", softplus(x))
print("Softmax:", softmax(x))
print("Mish:", mish(x))
