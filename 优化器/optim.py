import numpy as np

# ----------------------- 1. SGD -----------------------
class SGD:
    def __init__(self, lr=0.01, momentum=0, weight_decay=0):
        self.lr = lr
        self.mu = momentum
        self.wd = weight_decay
        self.v = None   # 动量缓存

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(w) for w in params]
        for i, (w, g) in enumerate(zip(params, grads)):
            
            g = g + self.wd * w                # L2正则（求导后）
            self.v[i] = self.mu * self.v[i] - self.lr * g
            w += self.v[i]

# ----------------------- 2. Adam -----------------------
class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.m = None   # 一阶动量
        self.v = None   # 二阶动量
        self.t = 0      # 时间步

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(w) for w in params]
            self.v = [np.zeros_like(w) for w in params]
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t)
        for i, (w, g) in enumerate(zip(params, grads)):
            g = g + self.wd * w
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            w -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

# ----------------------- 3. RMSprop -----------------------
class RMSprop:
    def __init__(self, lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=0):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.wd = weight_decay
        self.v = None   # 二阶动量缓存

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(w) for w in params]
        for i, (w, g) in enumerate(zip(params, grads)):
            g = g + self.wd * w
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * g * g
            w -= self.lr * g / (np.sqrt(self.v[i]) + self.eps)

# ----------------------- 4. Adagrad -----------------------
class Adagrad:
    def __init__(self, lr=1e-2, eps=1e-10, weight_decay=0):
        self.lr = lr
        self.eps = eps
        self.wd = weight_decay
        self.v = None   # 梯度平方累加

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(w) for w in params]
        for i, (w, g) in enumerate(zip(params, grads)):
            g = g + self.wd * w
            self.v[i] += g * g
            w -= self.lr * g / (np.sqrt(self.v[i]) + self.eps)

# ----------------------- 使用示例 -----------------------
if __name__ == "__main__":
    # 构造假数据
    w1 = np.random.randn(10, 20)
    w2 = np.random.randn(20)
    g1 = np.random.randn(*w1.shape)
    g2 = np.random.randn(*w2.shape)

    opts = {
        "SGD":      SGD(lr=0.1, momentum=0.9),
        "Adam":     Adam(lr=1e-3),
        "RMSprop":  RMSprop(lr=1e-3),
        "Adagrad":  Adagrad(lr=0.01),
    }

    for name, opt in opts.items():
        # 深拷贝，防止同一份参数被连续更新
        params = [w.copy() for w in [w1, w2]]
        grads  = [g.copy() for g in [g1, g2]]
        opt.update(params, grads)
        print(name, "updated. w1[0,0] =", params[0][0, 0])