import numpy as np

#线性回归
class LR:
	def __init__(self, lr=0.01, iter=1000):
		self.lr = lr
		self.iter = iter
		self.weights = None
		self.bias = None
	def fit(self, X, y):
		num_samples, num_features = X.shape
		self.weights = np.zeros(num_features)
		self.bias = 0
		for _ in range(self.iter):
			y_pred = np.dot(X, self.weights) + self.bias

			# MSE的梯度
			d_w = (1 / num_samples) * np.dot(X.T, y_pred - y)
			d_b = (1 / num_samples) * np.sum(y_pred - y)
			self.weights -= self.lr * d_w
			self.bias -= self.lr * d_b

	def predict(self, X):
		y_pred = np.dot(X, self.weights) + self.bias
		return y_pred


# 测试 main 函数
if __name__ == "__main__":
    # 生成测试数据（简单的线性关系 y = 3x + 5）	
	np.random.seed(42)
	X = np.random.rand(100, 3) * 10  # 100个样本，每个样本有3个特征
	true_weights = np.array([3, -2, 5])  # 真实权重
	y = X @ true_weights + 5 + np.random.randn(100) * 2  # 添加噪声

	# 训练模型
	model = LR(lr=0.01, iter=1000)
	model.fit(X, y)

	# 预测
	X_test = np.array([[1.5, 2.3, 3.1], [3.2, 1.1, 0.5]])  # 测试数据（每行3个特征）
	y_pred = model.predict(X_test)

	print("预测结果:\n", y_pred)
	print("训练得到的权重:", model.weights)
	print("训练得到的偏置:", model.bias)