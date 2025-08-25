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



if __name__ == '__main__':

    y = np.array([1,0,0,0,1,0,1,0,])
    pred = np.array([0.80001, 0.8000000000000001, 0.8, 0.8,0.8,0.8,0.8,0.8])


    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    print("-----sklearn:",auc(fpr, tpr))
    print("-----auc_calculate脚本:",auc_calculate(y,pred))
    print("-----AUC脚本:",AUC(y,pred))