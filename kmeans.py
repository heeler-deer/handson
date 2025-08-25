import numpy as np

def kmeans(X,K,max_iters=100):
    m,n=X.shape
    centroids = X[np.random.choice(m,K,replace=False)]
    pre_centroids=None
    labels=np.zeros(m)
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels=np.argmin(distances,axis=1)
    for k in range(K):
        clusters=X[labels==k]
        centroids[k]=np.mean(clusters)
        if centroids==pre_centroids:
            break
        pre_centroids=centroids
    return labels,centroids