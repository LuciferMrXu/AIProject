import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet     # 将测试点点拉伸与样本总数相同，对应相减
    sqDiffmat = diffMat**2
    sqDistances = sqDiffmat.sum(axis = 1)
    distances = sqDistances**0.5     # 测试点与其他所有样本的距离矩阵
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    print(sortedClassCount)
    return sortedClassCount[0][0]
if __name__ == '__main__':
    group,labels = createDataSet()
    res = classify0([0.2,0.3],group,labels,3)
    print(res)
    