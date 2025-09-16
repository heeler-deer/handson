import numpy as np

def dropout(x, rate, train=True):
    if train:
        # inverted dropout：训练时直接放大
        mask = np.random.binomial(1, 1-rate, size=x.shape) / (1-rate)
        return x * mask
    else:
        # 测试时无需任何缩放
        return x

def forward(x, w1, b1, w2, b2, rate, train=True):
    h1 = np.maximum(0, np.dot(w1, x) + b1)
    h1 = dropout(h1, rate, train=train)
    h2 = np.maximum(0, np.dot(w2, h1) + b2)
    h2 = dropout(h2, rate, train=train)
    return h2