import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1. 定义通用的塔模型
class TowerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TowerModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# 2. 定义双塔模型
class TwoTowerModel(nn.Module):
    def __init__(self, user_input_dim, item_input_dim, hidden_dim, embedding_dim):
        super(TwoTowerModel, self).__init__()
        self.user_tower = TowerModel(user_input_dim, hidden_dim, embedding_dim)
        self.item_tower = TowerModel(item_input_dim, hidden_dim, embedding_dim)

    def forward(self, user_features, item_features):
        user_embedding = self.user_tower(user_features)
        item_embedding = self.item_tower(item_features)
        
        # L2归一化，使得点积等价于余弦相似度
        user_embedding = F.normalize(user_embedding, p=2, dim=1)
        item_embedding = F.normalize(item_embedding, p=2, dim=1)
        
        return user_embedding, item_embedding

# 3. 训练和使用示例
if __name__ == '__main__':
    # --- 模型超参数 ---
    USER_FEATURE_DIM = 64
    ITEM_FEATURE_DIM = 32
    TOWER_HIDDEN_DIM = 128
    EMBEDDING_DIM = 64  # 用户和物料向量的最终维度
    
    # --- 训练参数 ---
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    EPOCHS = 5

    # --- 实例化模型 ---
    model = TwoTowerModel(
        user_input_dim=USER_FEATURE_DIM,
        item_input_dim=ITEM_FEATURE_DIM,
        hidden_dim=TOWER_HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("模型结构:")
    print(model)
    print("-" * 40)
    
    # --- 训练过程 ---
    print("--- 开始训练 ---")
    for epoch in range(EPOCHS):
        model.train()
        
        # --- 生成模拟数据 (一个Batch) ---
        user_inputs = torch.randn(BATCH_SIZE, USER_FEATURE_DIM)
        positive_item_inputs = torch.randn(BATCH_SIZE, ITEM_FEATURE_DIM)
        
        # --- 前向传播 ---
        user_embeddings, item_embeddings = model(user_inputs, positive_item_inputs)
        
        # --- 计算损失 (使用In-Batch Negatives) ---
        # 1. 计算批内所有用户和物料两两之间的相似度得分
        # user_embeddings: (BATCH_SIZE, EMBEDDING_DIM)
        # item_embeddings: (BATCH_SIZE, EMBEDDING_DIM)
        # logits: (BATCH_SIZE, BATCH_SIZE)
        logits = torch.matmul(user_embeddings, item_embeddings.T)
        
        # 2. 创建标签。对角线上的元素是正样本对(user_i, item_i)，所以标签是1。
        # 其余都是负样本对(user_i, item_j)，标签是0。
        # 我们使用一个简单的技巧，创建一个从0到BATCH_SIZE-1的序列作为标签。
        labels = torch.arange(BATCH_SIZE)
        
        # 3. 使用交叉熵损失函数
        # 它会计算Softmax(logits)和真实标签之间的损失。
        # 对于第i行，它促使模型在第i列（正样本）上输出最高的概率。
        loss = F.cross_entropy(logits, labels)
        
        # --- 反向传播和优化 ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    print("--- 训练完成 ---")
    print("-" * 40)
    
    # --- 推理和应用示例 ---
    print("--- 开始推理 ---")
    model.eval()
    with torch.no_grad():
        # 1. 假设我们有5个待测试的用户
        test_user_features = torch.randn(5, USER_FEATURE_DIM)
        # 2. 假设我们的物料库中有1000个物料
        item_pool_features = torch.randn(1000, ITEM_FEATURE_DIM)

        # 3. 离线计算所有物料的Embedding (现实中这一步会存储起来)
        all_item_embeddings = model.item_tower(item_pool_features)
        all_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)

        # 4. 线上服务：当一个用户请求到来时，实时计算用户Embedding
        test_user_embedding = model.user_tower(test_user_features[0].unsqueeze(0)) # 以第一个用户为例
        test_user_embedding = F.normalize(test_user_embedding, p=2, dim=1)

        # 5. 计算该用户与物料库中所有物料的相似度得分
        scores = torch.matmul(test_user_embedding, all_item_embeddings.T)
        
        # 6. 排序并获取Top-K推荐结果
        top_k = 5
        top_scores, top_indices = torch.topk(scores.squeeze(0), k=top_k)

        print(f"为用户 0 推荐的前 {top_k} 个物料:")
        for i in range(top_k):
            print(f"  - 物料索引: {top_indices[i].item()}, 相似度得分: {top_scores[i].item():.4f}")