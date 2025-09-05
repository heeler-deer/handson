import torch
import torch.nn as nn
import torch.optim as optim

class DeepFM(nn.Module):
    def __init__(self, num_features, num_fields, embedding_dim, hidden_dims):
        """
        初始化 DeepFM 模型
        """
        super(DeepFM, self).__init__()
        self.num_features = num_features
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        
        # --- Embedding 层 ---
        # 所有稀疏特征共享一个大的Embedding表
        self.embedding = nn.Embedding(num_features, embedding_dim)
        # 对每个特征的Embedding进行初始化，有助于模型收敛
        nn.init.xavier_uniform_(self.embedding.weight)

        # --- FM 部分 ---
        # 1. 一阶部分 (Linear Part)
        # 每个特征对应一个权重
        self.fm_linear = nn.Embedding(num_features, 1) 
        
        # 2. 二阶部分 (Interaction Part) 的计算在forward中实现
        
        # --- Deep 部分 ---
        # 1. 输入层
        # 输入维度是所有特征域的Embedding向量拼接起来的维度
        deep_input_dim = num_fields * embedding_dim
        
        # 2. 隐藏层和输出层
        layers = []
        input_dim = deep_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) # 加入BatchNorm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3)) # 加入Dropout
            input_dim = h_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, 1))
        
        self.deep_mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        前向传播
        :param x: 输入的稀疏特征索引, 形状为 (batch_size, num_fields)
        """
        # --- Embedding Lookup ---
        # (batch_size, num_fields) -> (batch_size, num_fields, embedding_dim)
        embeddings = self.embedding(x)

        # --- FM 部分计算 ---
        # 1. 一阶部分
        # (batch_size, num_fields) -> (batch_size, num_fields, 1)
        linear_part = self.fm_linear(x)
        # (batch_size, num_fields, 1) -> (batch_size, 1)
        fm_first_order = torch.sum(linear_part, dim=1)
        
        # 2. 二阶部分
        # 使用数学公式简化计算: 0.5 * [ (sum(v_i * x_i))^2 - sum((v_i * x_i)^2) ]
        # 在类别型特征场景下，每个域只有一个值为1，其他为0，所以x_i=1
        
        # (batch_size, num_fields, embedding_dim) -> (batch_size, embedding_dim)
        sum_of_square = torch.sum(embeddings, dim=1).pow(2) 
        
        # (batch_size, num_fields, embedding_dim) -> (batch_size, num_fields, embedding_dim)
        square_of_sum = embeddings.pow(2).sum(dim=1)
        
        fm_second_order = 0.5 * torch.sum(sum_of_square - square_of_sum, dim=1, keepdim=True)
        
        # --- Deep 部分计算 ---
        # 1. 准备输入
        # (batch_size, num_fields, embedding_dim) -> (batch_size, num_fields * embedding_dim)
        deep_input = embeddings.view(x.size(0), -1)
        
        # 2. MLP计算
        deep_output = self.deep_mlp(deep_input)
        
        # --- 最终输出 ---
        # 将三部分相加
        output = fm_first_order + fm_second_order + deep_output
        
        # 通过 Sigmoid 函数得到最终的CTR预测值
        return torch.sigmoid(output)

# --- 训练和使用示例 ---
if __name__ == '__main__':
    # --- 1. 定义模型和数据超参数 ---
    
    # 假设我们有3个特征域: user_id, item_id, gender
    NUM_FIELDS = 3
    
    # 假设 user_id 有1000个类别, item_id 有500个, gender 有2个
    # 特征索引需要进行偏移量处理，以构建一个全局的特征字典
    # user_id: 0-999
    # item_id: 1000-1499
    # gender: 1500-1501
    NUM_FEATURES = 1000 + 500 + 2 
    
    EMBEDDING_DIM = 16
    HIDDEN_DIMS = [128, 64] # Deep部分的MLP隐藏层
    
    # 训练参数
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 10

    # --- 2. 实例化模型 ---
    model = DeepFM(NUM_FEATURES, NUM_FIELDS, EMBEDDING_DIM, HIDDEN_DIMS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCELoss() # 二元交叉熵损失

    print("模型结构:")
    print(model)
    print("-" * 50)
    
    # --- 3. 训练过程 ---
    print("--- 开始训练 ---")
    for epoch in range(EPOCHS):
        model.train()
        
        # --- 生成模拟数据 (一个Batch) ---
        # 在真实场景中，你会从数据加载器中获取数据
        # 输入x是特征的全局索引
        user_ids = torch.randint(0, 1000, (BATCH_SIZE,))
        item_ids = torch.randint(1000, 1500, (BATCH_SIZE,))
        genders = torch.randint(1500, 1502, (BATCH_SIZE,))
        
        # 将特征域拼接起来
        # (BATCH_SIZE, NUM_FIELDS)
        x_batch = torch.stack([user_ids, item_ids, genders], dim=1)
        
        # 随机生成标签 (0或1)
        y_batch = torch.randint(0, 2, (BATCH_SIZE, 1)).float()
        
        # --- 前向传播和计算损失 ---
        y_pred = model(x_batch)
        loss = loss_function(y_pred, y_batch)
        
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
        test_user = torch.LongTensor([[50]])      # 用户ID为50
        test_item = torch.LongTensor([[1200]])    # 物品ID为200 (全局索引为1200)
        test_gender = torch.LongTensor([[1501]]) # 性别为1 (全局索引为1501)
        
        test_x = torch.cat([test_user, test_item, test_gender], dim=1)
        
        prediction = model(test_x)
        
        print(f"输入特征索引: {test_x.numpy()}")
        print(f"预测的点击率 (CTR): {prediction.item():.4f}")