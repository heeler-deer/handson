#
# 计算原理：所有的样本对中被正确排序的样本对（正类排在负类前面）的比例。
# 假设正样本M，负样本N，则正负样本对 总共有M*N中情况
# 按照预测值由小到大排序，对每一个正样本计算其正确排序的负样本（在其前面的负样本）个数，除以M*N，即为auc


import numpy as np  
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# 当 n_bins=100 时，如果预测值的范围（即 max - min）小于 0.01，那么该方法可能无法正确计算 AUC
def auc_calculate(labels,preds,n_bins=100):
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    total_case = postive_len * negative_len
    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i]/bin_width)
        if labels[i]==1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i]*accumulated_neg + pos_histogram[i]*neg_histogram[i]*0.5)
        accumulated_neg += neg_histogram[i]

    return satisfied_pair / float(total_case)

'''
ＡＵＣ的含义就是所有穷举所有的正负样本对，如果正样本的预测概率大于负样本的预测概率，那么就＋１；
如果如果正样本的预测概率等于负样本的预测概率，那么就＋0.5,　
如果正样本的预测概率小于负样本的预测概率，那么就＋０；
最后把统计处理的个数除以Ｍ×Ｎ就得到我们的ＡＵＣ
'''
def AUC(label, pre):
    pos = [i for i in range(len(label)) if label[i] == 1]
    neg = [i for i in range(len(label)) if label[i] == 0]
    auc = 0
    for i in pos:
        for j in neg:
            if pre[i] > pre[j]:
                auc += 1
            elif pre[i] == pre[j]:
                auc += 0.5
    return auc / (len(pos)*len(neg))





def fast_auc(label, pre):
    """
    使用排序优化的方法高效并正确地计算AUC，能正确处理预测分数相同的情况。
    时间复杂度为 O(K log K)，其中 K 是总样本数。
    """
    # 1. 将预测分数和真实标签打包并排序
    pairs = sorted(zip(pre, label), key=lambda x: x[0])

    # 2. 计算正样本和负样本的总数
    pos_num = sum(1 for _, l in pairs if l == 1)
    neg_num = len(label) - pos_num

    # 处理边界情况
    if pos_num == 0 or neg_num == 0:
        return 1.0

    # 3. 遍历排序后的列表，高效地计算分子
    #    分子是所有 (正样本，负样本) 对中，满足
    #    score(正) > score(负) 的对数 + 0.5 * (score(正) == score(负) 的对数)
    auc_numerator = 0.0
    neg_seen_count = 0  # 记录到目前为止已经遇到的负样本数量
    
    i = 0
    n = len(pairs)
    while i < n:
        # 找到当前分数相同的所有样本块
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        
        # ----- 处理这个分数相同的块 (从索引 i 到 j-1) -----
        
        # 统计这个块中有多少个正样本和负样本
        pos_in_block = sum(1 for k in range(i, j) if pairs[k][1] == 1)
        neg_in_block = (j - i) - pos_in_block
        
        # 第一部分贡献：
        # 这个块里的所有正样本，其分数都 > 在此之前遇到的所有负样本
        # 这贡献了 pos_in_block * neg_seen_count 个“+1”对
        auc_numerator += pos_in_block * neg_seen_count
        
        # 第二部分贡献：
        # 这个块里的所有正样本，其分数都 == 这个块里的所有负样本
        # 这贡献了 pos_in_block * neg_in_block 个“+0.5”对
        auc_numerator += pos_in_block * neg_in_block * 0.5
        
        # 更新已遇到的负样本总数
        neg_seen_count += neg_in_block
        
        # 移动索引到下一个块的开始
        i = j
        
    return auc_numerator / (pos_num * neg_num)








if __name__ == '__main__':

    y = np.array([1,0,0,0,1,0,1,0,])
    pred = np.array([0.80001, 0.8000000000000001, 0.8, 0.7,0.8,0.3,0.8,0.8])


    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    print("-----sklearn:",auc(fpr, tpr))
    print("-----auc_calculate脚本:",auc_calculate(y,pred))
    print("-----AUC脚本:",AUC(y,pred))
    print("---fastauc",fast_auc(y,pred))