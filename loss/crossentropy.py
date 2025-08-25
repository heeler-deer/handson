import numpy as np

def categorical_cross_entropy_loss(y_true, y_pred):
    """
    计算多分类问题的交叉熵损失

    参数：
    y_true (np.ndarray): 真实标签，形状为 (n_samples, n_classes)
    y_pred (np.ndarray): 预测概率，形状为 (n_samples, n_classes)

    返回：
    float: 交叉熵损失
    """
    epsilon = 1e-15  # 防止 log(0) 的情况
    y_pred=np.clip(y_pred,epsilon,1-epsilon)
    loss=-np.mean(np.sum(y_true*np.log(y_pred),axis=1))
    return loss