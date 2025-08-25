import numpy as np

def distance(x1, x2):
	return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
	def __init__(self, k=3):
		self.k = k

	def fit(self, X, y):
		self.X = X
		self.y = y

	def predict(self, X):
		y_pred = [self._predict(x) for x in X]
		return y_pred

	def _predict(self, x):
		distances = [distance(x, x_train) for x_train in self.X]
		k_indices = np.argsort(distances)[:self.k]
		k_labels = [self.y[i] for i in k_indices]
		y = np.bincount(k_labels).argmax()
		return y


## 并行计算
def knn_no_loops(data, query, k):
    """
    Perform a k-nearest neighbor search without using explicit loops.

    Args:
    data (numpy.ndarray): The dataset to search against, where each row is a data point.
    query (numpy.ndarray): The query point, as a 1D numpy array.
    k (int): The number of nearest neighbors to return.

    Returns:
    numpy.ndarray: The indices of the k nearest neighbors.
    """
    # 计算差值
    diff = data - query
    # 计算欧氏距离的平方
    dist_squared = np.sum(diff ** 2, axis=1)
    # 获取最小的k个距离的索引
    nearest_neighbors = np.argsort(dist_squared)[:k]
    return nearest_neighbors




X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]])
y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([[5, 5]])

knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(predictions)  # 输出 [1]