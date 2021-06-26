'''
Created on Aug 19, 2016

@author: tangyudi
'''

import numpy as np
import os
from knn_brute import  *

def img2vector(filename):
    returnVect = np.zeros((1,1024))      # 将数据转化成行向量
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        if lineStr=='\n':
            pass
        else:
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])

    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('./datas/trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('./datas/trainingDigits/%s' % fileNameStr)

    testFileList = os.listdir('./datas/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('./datas/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

if __name__ == '__main__':
    handwritingClassTest()