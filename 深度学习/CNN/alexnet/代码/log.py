#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/7 16:36
# @Author  : wufeiyu
# @Site    : 
# @File    : log.py
# @Software: PyCharm Community Edition

# 加载数据
import numpy as np
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return labelMat, dataMat


# loadDataSet()


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMat)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))  # n行1列
    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = labelMat - h
        weights = weights + alpha * dataMat.transpose() * error
    return weights


labelMat, dataMat = loadDataSet()
weights = gradAscent(dataMat, labelMat)


def result():
    dataMat2 = np.array([1.0, -1.395634	,4.662541])
    prob = sigmoid(dataMat2 * weights)
    print(prob)
    if prob > 0.5:
        return 1.0
    else:
        return 0


print(result())
