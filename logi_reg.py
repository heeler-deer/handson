import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, iterr=1000):
        self.lr = lr  # 学习率
        self.iter = iterr  # 训练迭代次数
        self.weights = None  # 权重
        self.bias = None  # 偏置

    def sigmoid(self, z):
        """计算 Sigmoid 激活函数"""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """训练逻辑回归模型"""
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.iter):
            # 计算线性部分
            linear_model = np.dot(X, self.weights) + self.bias
            # 计算Sigmoid输出
            y_pred = self.sigmoid(linear_model)

            # 计算梯度
            d_w = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            d_b = (1 / num_samples) * np.sum(y_pred - y)

            # 更新参数
            self.weights -= self.lr * d_w
            self.bias -= self.lr * d_b

    def predict_proba(self, X):
        """返回预测的概率"""
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """返回最终分类结果（0 或 1）"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
