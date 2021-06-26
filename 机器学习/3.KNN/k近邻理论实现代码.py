import numpy as np
import operator
import pandas as pd


def classify0(inX,dataSet,labels,k):    # KNN原理代码
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet   # 将测试点点拉伸与样本总数相同，对应相减
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distance = sqDistances **0.5               # 测试点与其他所有样本的距离矩阵
    sortedDistIndicies = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):   # 读取数据
    fr=pd.read_csv(filename,header=None,sep='\t')
    returnMat=fr.iloc[:,0:3]
    classLabelVector=fr.iloc[:,-1]
    return returnMat,classLabelVector


def autoNorm(dataSet):       # 归一化操作，使不同特征具有相同权重的值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet / np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():   # 划分训练集与测试集
    hoRatio = 0.1    # 划分百分之十的样本为测试集
    datingDataMat,datingLabels = file2matrix('./datas/datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)

    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)

    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat.values[i,:],normMat.values[numTestVecs:m,:],datingLabels.values[numTestVecs:m],4)
        print('the classifier came back with: %d , the real answer is : %d'%(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print('total result is : %f' %(errorCount / float(numTestVecs)))    # 误差率

def classifyperson():    # 模型预测
    resultList = ['not at all', 'in small does','in large does']
    input_man= [20000,10,5]
    datingDataMat,datingLabels = file2matrix('./datas/datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat) 
    result = classify0((input_man - minVals)/ranges,normMat,datingLabels,3)
    print('you will probably like this person:' ,  resultList[result-1])

if __name__ == '__main__':
    datingClassTest()
    classifyperson()