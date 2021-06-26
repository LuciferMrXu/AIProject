import numpy as np

def loadSimData():
    datMat = np.matrix([[1.0,2.1],
                       [2.0,1.1],
                       [1.3,1.0],
                       [1.0,1.0],
                       [2.0,1.0]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

def stumpClassify(datMatrix,dimen,threshval,threshIneq):
    retArray = np.ones((np.shape(datMatrix)[0],1))
    if threshIneq == 'lt':
        # 小于等于阈值v时取-1
        retArray[datMatrix[:,dimen] <= threshval] = -1.0
    else:
        # 大于阈值v时取-1
        retArray[datMatrix[:,dimen] > threshval] = -1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)   # m代表有几条样本，n代表样本有几个特征
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin) / numSteps    # 步长
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshval = (rangeMin + float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshval,inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0  # 错误率
                weightArr = D.T*errArr   # 权重
                
                if weightArr < minError:
                    minError = weightArr
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i  # 返回最好维度
                    bestStump['thresh'] = threshval   # 返回最好的阈值
                    bestStump['ineq'] = inequal   # 返回最优切分方式

    return bestStump,minError,bestClassEst


def adaBoostTrainDs(dataArr,classLabels,numIt = 40):
    werkClassArr = []    # 把结果储存到弱分类器中
    m = np.shape(dataArr)[0]    # 找多少个样本点
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))   # 初始化
    for i in range(numIt):
        bestStump,error,classEst = buildStump(datMat,classLabels,D)

        print('D:',D.T)   # 权重
        # 先将1e-16变成list形式进行拼接，注意输入为一个tuple
        new_error=np.concatenate((error,[[1e-16]]))

        alpha = float(0.5*np.log((1.0-error)/np.max(new_error)))
        bestStump['alpha'] = alpha
        werkClassArr.append(bestStump)

        print('classEst',classEst)   # 预测结果

        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
        D = np.multiply(D,np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst

        print('aggClassEst',aggClassEst.T)   # 带权重的预测结果

        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        print('total error:',errorRate)
        if errorRate == 0.0:
            break
    return werkClassArr
if __name__ == '__main__':
    datMat,classLabels = loadSimData()
    adaBoostTrainDs(datMat,classLabels,10)