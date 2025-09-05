import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class AttentionUnit(nn.Module):
    """
    注意力单元 (Activation Unit)
    """
    def __init__(self, embedding_dim, hidden_dims):
        super(AttentionUnit, self).__init__()
        
        # 注意力网络，通常是一个小型的MLP
        layers = []
        input_dim = embedding_dim * 4  # 输入是 candidate_ad, history_item, ad-hist, ad*hist
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.attention_net = nn.Sequential(*layers)
    
    def forward(self, candidate_embedding, history_embeddings, mask):
        # 扩展候选广告的维度以匹配历史序列
        # (batch_size, embedding_dim) -> (batch_size, seq_len, embedding_dim)
        seq_len = history_embeddings.size(1)
        candidate_expanded = candidate_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 拼接特征，作为注意力网络的输入
        # 原始论文中使用了 a, b, a-b, a*b 作为输入
        concat_features = torch.cat([
            candidate_expanded, 
            history_embeddings,
            candidate_expanded - history_embeddings,
            candidate_expanded * history_embeddings
        ], dim=-1)
        
        # (batch_size, seq_len, embedding_dim * 4) -> (batch_size, seq_len, 1)
        attention_scores = self.attention_net(concat_features)
        
        # 将分数形状变为 (batch_size, seq_len)
        attention_scores = attention_scores.squeeze(-1)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax得到权重
        # (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)
        # attention_weights: (batch_size, seq_len) -> (batch_size, seq_len, 1)
        # history_embeddings: (batch_size, seq_len, embedding_dim)
        # weighted_sum: (batch_size, embedding_dim)
        weighted_sum = torch.sum(history_embeddings * attention_weights.unsqueeze(-1), dim=1)
        
        return weighted_sum

class DIN(nn.Module):
    def __init__(self, num_items, embedding_dim, attention_hidden_dims, mlp_hidden_dims):
        super(DIN, self).__init__()
        
        # 物品的Embedding层 (包括候选物品和历史物品)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        
        # 注意力单元
        self.attention_unit = AttentionUnit(embedding_dim, attention_hidden_dims)
        
        # 最终的MLP预测层
        # 输入维度: 候选物品Emb + 用户兴趣Emb(Attention输出)
        mlp_input_dim = embedding_dim * 2
        
        layers = []
        for h_dim in mlp_hidden_dims:
            layers.append(nn.Linear(mlp_input_dim, h_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(mlp_hidden_dims[-1], 1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, candidate_item_id, history_item_ids):
        # 1. Embedding Lookup
        # (batch_size, 1, embedding_dim) -> (batch_size, embedding_dim)
        candidate_embedding = self.item_embedding(candidate_item_id).squeeze(1)
        # (batch_size, seq_len, embedding_dim)
        history_embeddings = self.item_embedding(history_item_ids)
        
        # 2. 创建掩码 (mask)
        # padding的ID为0，所以历史序列中ID为0的位置是无效的
        mask = (history_item_ids != 0)
        
        # 3. 通过Attention Unit计算用户兴趣向量
        user_interest_embedding = self.attention_unit(candidate_embedding, history_embeddings, mask)
        
        # 4. 拼接特征并送入MLP
        combined_features = torch.cat([candidate_embedding, user_interest_embedding], dim=-1)
        
        # 5. 得到最终输出
        output = self.mlp(combined_features)
        
        return torch.sigmoid(output)

# --- 训练和使用示例 ---
if __name__ == '__main__':
    # --- 1. 定义模型和数据超参数 ---
    NUM_ITEMS = 1001       # 物品总数 (0用作padding)
    EMBEDDING_DIM = 64
    ATTENTION_HIDDEN_DIMS = [80, 40]
    MLP_HIDDEN_DIMS = [200, 80]
    
    # 训练参数
    BATCH_SIZE = 64
    SEQ_LEN = 20           # 用户历史行为序列的最大长度
    LEARNING_RATE = 0.001
    EPOCHS = 10

    # --- 2. 实例化模型 ---
    model = DIN(NUM_ITEMS, EMBEDDING_DIM, ATTENTION_HIDDEN_DIMS, MLP_HIDDEN_DIMS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCELoss()

    print("模型结构:")
    print(model)
    print("-" * 50)
    
    # --- 3. 训练过程 ---
    print("--- 开始训练 ---")
    for epoch in range(EPOCHS):
        model.train()
        
        # --- 生成模拟数据 (一个Batch) ---
        # 候选物品ID
        candidate_ids = torch.randint(1, NUM_ITEMS, (BATCH_SIZE, 1))
        
        # 历史行为序列ID (包含padding)
        history_ids = torch.randint(0, NUM_ITEMS, (BATCH_SIZE, SEQ_LEN))
        # 随机设置一些padding (ID为0)
        for i in range(BATCH_SIZE):
            pad_len = torch.randint(0, SEQ_LEN//2, (1,)).item()
            if pad_len > 0:
                history_ids[i, -pad_len:] = 0
        
        # 随机生成标签
        labels = torch.randint(0, 2, (BATCH_SIZE, 1)).float()
        
        # --- 前向传播和计算损失 ---
        predictions = model(candidate_ids, history_ids)
        loss = loss_function(predictions, labels)
        
        # --- 反向传播和优化 ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    print("--- 训练完成 ---")
    print("-" * 50)

    # --- 4. 推理示例 ---
    print("--- 开始推理 ---")
    model.eval()
    with torch.no_grad():
        # 创建一个测试样本
        test_candidate = torch.LongTensor([[100]]) # 候选物品 100
        # 历史行为: [10, 25, 101, 300, 0, 0, ...] (后面是padding)
        test_history = torch.LongTensor([[10, 25, 101, 300] + [0]*(SEQ_LEN-4)])
        
        prediction = model(test_candidate, test_history)
        
        print(f"候选物品ID: {test_candidate.item()}")
        print(f"用户历史行为序列: {test_history.numpy().flatten()}")
        print(f"预测的点击率 (CTR): {prediction.item():.4f}")