#
#https://blog.csdn.net/comli_cn/article/details/129104938
#


def average_precision(y_true):
    """
    计算单个查询的平均准确率（Average Precision）
    """
    # 初始化变量
    total_relevant = 0
    precision_sum = 0.0
    for i, value in enumerate(y_true):
        if value == 1:
            total_relevant += 1
            precision_sum += total_relevant / (i + 1)
    if total_relevant == 0:
        return 0
    return precision_sum / total_relevant

def mean_average_precision(y_true_list):
    """
    计算多个查询的平均准确率的平均值（Mean Average Precision）
    """
    return sum(average_precision(y_true) for y_true in y_true_list) / len(y_true_list)

# 示例数据，每个查询的真实值（0代表不相关，1代表相关）
query_results = [
    [0, 1, 0, 1, 1],  # 第一个查询的真实值
    [1, 0, 0, 1, 0],  # 第二个查询的真实值
    # ... 更多查询的真实值
]

# 计算MAP
map_score = mean_average_precision(query_results)
print("Mean Average Precision (MAP):", map_score)


