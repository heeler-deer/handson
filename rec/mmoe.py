import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    专家网络
    一个简单的多层感知机 (MLP)
    """
    def __init__(self, input_dim, expert_dim):
        super(Expert, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, expert_dim)
        )

    def forward(self, x):
        return self.fc(x)

class Gate(nn.Module):
    """
    门控网络
    为每个任务生成对应专家的权重。
    """
    def __init__(self, input_dim, num_experts):
        super(Gate, self).__init__()
        # 使用一个简单的线性层后接Softmax来生成权重
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # 使用 Softmax 保证所有专家权重之和为1
        return F.softmax(self.fc(x), dim=1)

class Tower(nn.Module):
    """
    任务塔
    每个任务独有的网络，用于最终的预测。
    """
    def __init__(self, input_dim, tower_dim, output_dim):
        super(Tower, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, tower_dim),
            nn.ReLU(),
            nn.Linear(tower_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class MMoE(nn.Module):
    """
    MMoE 主模型
    """
    def __init__(self, input_dim, num_experts, expert_dim, num_tasks, tower_dim, output_dims):
        """
        初始化MMoE模型
        :param input_dim: 输入特征的维度
        :param num_experts: 专家的数量
        :param expert_dim: 每个专家网络的输出维度
        :param num_tasks: 任务的数量
        :param tower_dim: 每个任务塔的隐藏层维度
        :param output_dims: 一个列表，包含每个任务的输出维度 (例如，二分类任务为1)
        """
        super(MMoE, self).__init__()
        
        # 检查参数
        assert num_tasks == len(output_dims), "任务数量必须与输出维度列表的长度一致"

        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # 专家网络
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_dim) for _ in range(num_experts)
        ])

        # 门控网络
        self.gates = nn.ModuleList([
            Gate(input_dim, num_experts) for _ in range(num_tasks)
        ])

        # 任务塔
        self.towers = nn.ModuleList([
            Tower(expert_dim, tower_dim, output_dims[i]) for i in range(num_tasks)
        ])

    def forward(self, x):
        # 1. 获取所有专家的输出
        expert_outputs = [expert(x) for expert in self.experts]
        # 将列表转换为张量，形状为 (batch_size, num_experts, expert_dim)
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # 2. 获取每个任务的门控权重并进行加权求和
        task_inputs = []
        for i in range(self.num_tasks):
            # 获取门控权重，形状为 (batch_size, num_experts)
            gate_weights = self.gates[i](x)
            # 增加一个维度以进行广播乘法，形状变为 (batch_size, num_experts, 1)
            gate_weights = gate_weights.unsqueeze(-1)

            # 加权求和
            # expert_outputs: (batch_size, num_experts, expert_dim)
            # gate_weights:   (batch_size, num_experts, 1)
            # weighted_experts: (batch_size, num_experts, expert_dim)
            # task_input:       (batch_size, expert_dim)
            task_input = torch.sum(expert_outputs * gate_weights, dim=1)
            task_inputs.append(task_input)

        # 3. 将加权后的专家输出送入各自的任务塔
        task_outputs = []
        for i in range(self.num_tasks):
            output = self.towers[i](task_inputs[i])
            task_outputs.append(output)

        return task_outputs
    
    
    
if __name__ == '__main__':
    # --- 1. 定义模型超参数 ---
    INPUT_DIM = 64      # 输入特征维度
    NUM_EXPERTS = 8     # 专家数量
    EXPERT_DIM = 32     # 每个专家的输出维度
    NUM_TASKS = 2       # 任务数量 (CTR, CVR)
    TOWER_DIM = 16      # 任务塔的隐藏层维度
    OUTPUT_DIMS = [1, 1]  # 每个任务的输出维度 (二分类都是1)

    # --- 2. 生成模拟数据 ---
    BATCH_SIZE = 128
    # 随机生成输入特征
    inputs = torch.randn(BATCH_SIZE, INPUT_DIM)
    
    # 随机生成标签（0或1）
    # 任务1: CTR 标签
    labels_ctr = torch.randint(0, 2, (BATCH_SIZE, 1)).float()
    # 任务2: CVR 标签
    labels_cvr = torch.randint(0, 2, (BATCH_SIZE, 1)).float()
    
    labels = [labels_ctr, labels_cvr]

    # --- 3. 实例化模型 ---
    mmoe_model = MMoE(
        input_dim=INPUT_DIM,
        num_experts=NUM_EXPERTS,
        expert_dim=EXPERT_DIM,
        num_tasks=NUM_TASKS,
        tower_dim=TOWER_DIM,
        output_dims=OUTPUT_DIMS
    )
    
    # --- 4. 定义损失函数和优化器 ---
    # 对每个任务使用二元交叉熵损失
    loss_functions = [nn.BCEWithLogitsLoss() for _ in range(NUM_TASKS)]
    optimizer = torch.optim.Adam(mmoe_model.parameters(), lr=0.001)

    # --- 5. 训练步骤 ---
    print("--- 开始训练 ---")
    mmoe_model.train()
    optimizer.zero_grad()
    
    # 获取模型输出
    task_predictions = mmoe_model(inputs)
    
    # 计算每个任务的损失
    loss_task1 = loss_functions[0](task_predictions[0], labels[0])
    loss_task2 = loss_functions[1](task_predictions[1], labels[1])
    
    # 总损失是所有任务损失的加权和（这里简单相加）
    total_loss = loss_task1 + loss_task2
    
    # 反向传播和参数更新
    total_loss.backward()
    optimizer.step()
    
    print(f"任务1 (CTR) 的损失: {loss_task1.item():.4f}")
    print(f"任务2 (CVR) 的损失: {loss_task2.item():.4f}")
    print(f"总损失: {total_loss.item():.4f}")
    print("--- 训练完成 ---")

    # --- 6. 推理步骤 ---
    print("\n--- 开始推理 ---")
    mmoe_model.eval()
    with torch.no_grad():
        test_inputs = torch.randn(5, INPUT_DIM)
        predictions = mmoe_model(test_inputs)
        
        # 使用Sigmoid函数将logits转换为概率
        pred_probs_task1 = torch.sigmoid(predictions[0])
        pred_probs_task2 = torch.sigmoid(predictions[1])
        
        print("测试样本的输入形状:", test_inputs.shape)
        print("\n任务1 (CTR) 的预测概率:")
        print(pred_probs_task1)
        print("\n任务2 (CVR) 的预测概率:")
        print(pred_probs_task2)