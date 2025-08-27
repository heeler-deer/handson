##
## https://blog.csdn.net/comli_cn/article/details/129104938
##

import numpy as np

def dcg_at_k(r, k):
    """
    计算DCG@K（Discounted Cumulative Gain at K）
    r: relevance scores
    k: 排名前K个文档
    """
    r = np.asfarray(r)[:k]  # 将r转换为浮点数类型，仅保留前k个元素
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r, k):
    """
    计算NDCG@K（Normalized Discounted Cumulative Gain at K）
    r: relevance scores
    k: 排名前K个文档
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max

# 示例数据，每个查询的相关性得分
query_results = [
    [3, 2, 3, 0, 1],  # 第一个查询的相关性得分
    [2, 1, 2, 3, 2],  # 第二个查询的相关性得分
    # ... 更多查询的相关性得分
]

# 计算NDCG@K
k = 3  # 假设计算前3个文档的NDCG
ndcg_scores = [ndcg_at_k(r, k) for r in query_results]
mean_ndcg = np.mean(ndcg_scores)
print(f"Mean NDCG@{k}: {mean_ndcg}")


