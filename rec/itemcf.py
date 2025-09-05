import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def item_based_collaborative_filtering():
    """
    一个完整的基于物品的协同过滤实现示例
    """
    # --- 1. 准备数据：创建一个用户-物品评分矩阵 ---
    # 现实世界中，这些数据来自数据库或日志文件。
    # 这里我们创建一个简单的数据集：
    # - 5个用户 (User1 到 User5)
    # - 6个物品 (ItemA 到 ItemF)
    # - 数值代表评分 (1-5)，NaN 代表用户未对该物品评分
    
    data = {
        'ItemA': [5, 4, np.nan, 1, np.nan],
        'ItemB': [4, 3, 5, 1, 2],
        'ItemC': [np.nan, 5, 4, np.nan, 1],
        'ItemD': [5, 4, np.nan, 2, np.nan],
        'ItemE': [1, np.nan, 2, 4, 5],
        'ItemF': [2, 1, 3, 5, 4],
    }
    users = ['User1', 'User2', 'User3', 'User4', 'User5']
    
    # 创建DataFrame作为我们的用户-物品评分矩阵
    utility_matrix = pd.DataFrame(data, index=users)
    print("1. 用户-物品评分矩阵 (Utility Matrix):")
    print(utility_matrix)
    print("-" * 40)

    # --- 2. 计算物品之间的相似度 ---
    # 为了计算物品相似度，我们需要处理缺失值(NaN)。
    # 一种常见的做法是用0或者用户的平均分填充，这里我们用0填充。
    # 注意：填充后的矩阵主要用于计算相似度，而不是用于最终的评分预测。
    utility_matrix_filled = utility_matrix.fillna(0)
    
    # scikit-learn的cosine_similarity需要 (n_samples, n_features) 格式的输入。
    # 在物品协同过滤中，每个物品是"sample"，每个用户是"feature"。
    # 所以我们需要转置矩阵，使其形状变为 (n_items, n_users)。
    item_vectors = utility_matrix_filled.T
    
    # 计算物品间的余弦相似度
    item_similarity_matrix = cosine_similarity(item_vectors)
    
    # 为了方便查看，将其转换为DataFrame
    item_similarity_df = pd.DataFrame(
        item_similarity_matrix,
        index=utility_matrix.columns,
        columns=utility_matrix.columns
    )
    
    print("2. 物品-物品相似度矩阵 (Item-Item Similarity Matrix):")
    print(item_similarity_df)
    print("-" * 40)
    
    # --- 3. 为指定用户生成推荐 ---
    target_user = 'User3'
    num_recommendations = 2

    print(f"3. 为 '{target_user}' 生成 {num_recommendations} 个推荐:")
    
    # 获取目标用户的所有评分
    user_ratings = utility_matrix.loc[target_user]
    
    # 找到用户已经评过分的物品
    rated_items = user_ratings.dropna().index
    print(f"\n'{target_user}' 已经评过分的物品: {list(rated_items)}")
    
    # 初始化一个字典来存储每个候选推荐物品的预测得分
    recommendation_scores = {}
    
    # 遍历用户评过分的每一个物品
    for item_rated in rated_items:
        # 获取该物品的评分
        rating = user_ratings[item_rated]
        
        # 获取与该物品相似的其他物品
        similar_items = item_similarity_df[item_rated]
        
        # 遍历相似物品，计算预测得分
        for similar_item, similarity_score in similar_items.items():
            # 跳过自身和相似度小于等于0的物品
            if similar_item == item_rated or similarity_score <= 0:
                continue
                
            # 预测得分的核心计算：
            # 某个物品的预测得分 = sum(相似物品的相似度 * 用户对该相似物品的评分) / sum(所有相似度)
            # 这里我们简化一下，直接用相似度 * 评分进行累加
            # 权重就是物品间的相似度
            weighted_score = similarity_score * rating
            
            # 将得分累加到候选推荐物品上
            if similar_item not in recommendation_scores:
                recommendation_scores[similar_item] = 0
            recommendation_scores[similar_item] += weighted_score

    # 过滤掉用户已经评过分的物品
    for item in rated_items:
        if item in recommendation_scores:
            del recommendation_scores[item]
            
    # 对推荐结果按分数从高到低排序
    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 获取前 N 个推荐
    top_n_recommendations = sorted_recommendations[:num_recommendations]

    print("\n生成的推荐列表 (物品: 预测得分):")
    if top_n_recommendations:
        for item, score in top_n_recommendations:
            print(f"- {item}: {score:.4f}")
    else:
        print("没有可推荐的物品。")

# 运行主函数
if __name__ == '__main__':
    item_based_collaborative_filtering()