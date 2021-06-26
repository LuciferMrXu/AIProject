'''
Created on Aug 19, 2016

@author: tangyudi
'''
import numpy as np
import pandas as pd
from numpy import tile
from knn_brute import classify0

def file2matrix(filename):
    fr=pd.read_csv(filename,header=None,sep='\t')
    returnMat=fr.iloc[:,:-1]
    classLabelVector=fr.iloc[:,-1]
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.10
    dataingDataSet,datingLabels = file2matrix('./datas/datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(dataingDataSet)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat.values[i,:],normMat.values[numTestVecs:m,:],datingLabels.values[numTestVecs:m],3)
        print('the classifier came back with : %d, the real answer is : %d'%(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print('the total test error rate is %f' %(errorCount/float(numTestVecs)))
def classifyPerson():
    resultList = ['not at all','in small does','in large does']
    datingMat,dataingLabel = file2matrix('./datas/datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingMat)
    inArr = np.array([50000,10,0.5])
    classifyResult = classify0((inArr - minVals)/ranges,normMat, dataingLabel, 5)
    print('you will probably like this person :' ,resultList[classifyResult - 1])
if __name__ == '__main__':
    datingClassTest()
    classifyPerson()
    
    
    