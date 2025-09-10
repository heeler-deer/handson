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
