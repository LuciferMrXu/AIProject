#_*_ coding:utf-8_*_
import numpy as np


def loadDataSet():
    postingList=[
        ['my','dog','has','flea','problems','help','please'],
        ['maybe','not','take','him','to','dog','park','stupid'],
        ['my','dalmation','is','so','cute','I','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr','licks','ate','my','steak','how','to','stop','him'],
        ['quit','buying','worthless','dog','food','stupid']
    ]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)    # 取出所有不重复的单词
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('该单词：%s不在词典中'%word)
    return returnVec


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/numTrainDocs
    p0Num=np.ones(numWords)
    p0Denom=2.0
    p1Num=np.ones(numWords)
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect=np.log(p1Num/p1Denom)          # 计算出现的单词在不同类别文章中的权重
    p0Vect = np.log(p0Num / p0Denom)      # 取log防止下值溢出
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1-pClass1)
    if p1>p0:               # 按照属于某个类别的概率大小进行分类
        return 1
    else:
        return 0

def testingNB(test):
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry=test
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    print('测试文章的类别为：%s'%classifyNB(thisDoc,p0V,p1V,pAb))

if __name__=='__main__':
    test=['stupid','my','dog']
    testingNB(test)