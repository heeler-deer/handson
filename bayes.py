#! 朴素贝叶斯分类器_西瓜集分类
import numpy as np
import math
import pandas as pd

#? 加载数据集
def loadDataSet():
    dataSet=[['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
             ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
             ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
             ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
             ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],            
             ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],               
             ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],                
             ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
 
             ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
             ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
             ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
             ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
             ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],  
             ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
             ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
             ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
             ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']]
    testSet= ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460] # 待测集
    labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率'] # 特征
    
    return dataSet, testSet, labels
 
#? 计算(不同类别中指定连续特征的)均值、标准差
def mean_std(feature, cla):
    dataSet, testSet, labels = loadDataSet()
    lst = [item[labels.index(feature)] for item in dataSet if item[-1]==cla]    # 类别为cla中指定特征feature组成的列表
    mean = round(np.mean(lst), 3)   # 均值
    std = round(np.std(lst), 3)     # 标准差
 
    return mean, std
 
#? 计算先验概率P(c)
def prior():
    dataSet = loadDataSet()[0]  # 载入数据集
    countG = 0  # 初始化好瓜数量
    countB = 0  # 初始化坏瓜数量
    countAll = len(dataSet)
 
    for item in dataSet:    # 统计好瓜个数
        if item[-1] == "好瓜":
            countG += 1
    for item in dataSet:    # 统计坏瓜个数
        if item[-1] == "坏瓜":
            countB += 1
    
    # 计算先验概率P(c)
    P_G = round(countG/countAll, 3)
    P_B = round(countB/countAll, 3)
 
    return P_G,P_B
 
#? 计算离散属性的条件概率P(xi|c)
def P(index, cla):
    dataSet, testSet, labels = loadDataSet()    # 载入数据集
    countG = 0  # 初始化好瓜数量
    countB = 0  # 初始化坏瓜数量
 
    for item in dataSet:    # 统计好瓜个数
        if item[-1] == "好瓜":
            countG += 1
    for item in dataSet:    # 统计坏瓜个数
        if item[-1] == "坏瓜":
            countB += 1
    
    lst = [item for item in dataSet if (item[-1] == cla) & (item[index] == testSet[index])] # lst为cla类中第index个属性上取值为xi的样本组成的集合
    P = round(len(lst)/(countG if cla=="好瓜" else countB), 3)  # 计算条件概率
 
    return P
 
#? 计算连续属性的条件概率p(xi|c)
def p():
    dataSet, testSet, labels = loadDataSet()    # 载入数据集
    denG_mean, denG_std = mean_std("密度", "好瓜")      # 好瓜密度的均值、标准差
    denB_mean, denB_std = mean_std("密度", "坏瓜")      # 坏瓜密度的均值、标准差
    sugG_mean, sugG_std = mean_std("含糖率", "好瓜")    # 好瓜含糖率的均值、标准差
    sugB_mean, sugB_std = mean_std("含糖率", "坏瓜")    # 坏瓜含糖率的均值、标准差
    # p(密度|好瓜)
    p_density_G = (1/(math.sqrt(2*math.pi)*denG_std))*np.exp(-(((testSet[labels.index("密度")]-denG_mean)**2)/(2*(denG_std**2))))
    p_density_G = round(p_density_G, 3)
    # p(密度|坏瓜)
    p_density_B = (1/(math.sqrt(2*math.pi)*denB_std))*np.exp(-(((testSet[labels.index("密度")]-denB_mean)**2)/(2*(denB_std**2))))
    p_density_B = round(p_density_B, 3)
    # p(含糖率|好瓜)
    p_sugar_G = (1/(math.sqrt(2*math.pi)*sugG_std))*np.exp(-(((testSet[labels.index("含糖率")]-sugG_mean)**2)/(2*(sugG_std**2))))
    p_sugar_G = round(p_sugar_G, 3)
    # p(含糖率|坏瓜)
    p_sugar_B = (1/(math.sqrt(2*math.pi)*sugB_std))*np.exp(-(((testSet[labels.index("含糖率")]-sugB_mean)**2)/(2*(sugB_std**2))))
    p_sugar_B = round(p_sugar_B, 3)
 
    return p_density_G, p_density_B, p_sugar_G, p_sugar_B
 
#? 预测后验概率P(c|xi)
def bayes():
    #? 计算类先验概率
    P_G, P_B = prior()
    #? 计算离散属性的条件概率
    P0_G = P(0, "好瓜") # P(青绿|好瓜)
    P0_B = P(0, "坏瓜") 
    P1_G = P(1, "好瓜") # P(蜷缩|好瓜)
    P1_B = P(1, "坏瓜") 
    P2_G = P(2, "好瓜") # P(浊响|好瓜)
    P2_B = P(2, "坏瓜") 
    P3_G = P(3, "好瓜") # P(清晰|好瓜)
    P3_B = P(3, "坏瓜") 
    P4_G = P(4, "好瓜") # P(凹陷|好瓜)
    P4_B = P(4, "坏瓜") 
    P5_G = P(5, "好瓜") # P(硬滑|好瓜)
    P5_B = P(5, "坏瓜") 
    #? 计算连续属性的条件概率
    p_density_G, p_density_B, p_sugar_G, p_sugar_B = p()
 
    #? 计算后验概率
    isGood = P_G * P0_G * P1_G * P2_G * P3_G * P4_G * P5_G * p_density_G * p_sugar_G    # 计算是好瓜的后验概率
    isBad = P_B * P0_B * P1_B * P2_B * P3_B * P4_B * P5_B * p_density_B * p_sugar_B     # 计算是坏瓜的后验概率
    '''print(P_G)
    print(P_B)
    print(P0_G)
    print(P0_B)
    print(P1_G)
    print(P1_B)
    print(P2_G)
    print(P2_B)
    print(P3_G)
    print(P3_B)
    print(P4_G)
    print(P4_B)
    print(P5_G)
    print(P5_B)
    print(p_density_G)
    print(p_density_B)
    print(p_sugar_G)
    print(p_sugar_B)'''
 
    return isGood,isBad
 
if __name__=='__main__':
    dataSet, testSet, labels = loadDataSet()
    testSet = [testSet]
    df = pd.DataFrame(testSet, columns=labels, index=[1])
    print("=======================待测样本========================")
    print(f"待测集:\n{df}")
 
    isGood, isBad = bayes()
    print("=======================后验概率========================")
    print("后验概率:")
    print(f"P(好瓜|xi) = {isGood}")
    print(f"P(好瓜|xi) = {isBad}")
    print("=======================预测结果========================")
    print("predict ---> 好瓜" if (isGood > isBad) else "predict ---> 坏瓜")