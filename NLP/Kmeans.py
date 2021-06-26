# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

def loadDataSet(fileName):      # 解析制表符分隔的浮点数的常规函数
    fr=np.genfromtxt(fileName, delimiter="\t", dtype=float)

    x1=fr[:,0]
    x2=fr[:,1]
    plt.figure(facecolor='w')
    plt.scatter(x1, x2, color='r', label='数据分布')
    plt.legend(loc='best')
    plt.title("K-Means", fontsize=20)
    plt.show()

    return fr


def distEclud(vecA, vecB):

    return np.sqrt(np.sum(np.power(vecA - vecB, 2))) #欧式距离公式



def randCent(dataSet, k):

    n = np.shape(dataSet)[1]

    centroids = np.mat(np.zeros((k,n)))    # 创建初始化矩阵

    for j in range(n):   #在每个维度的边界内创建随机簇

        minJ = min(dataSet[:,j]) 

        rangeJ = float(max(dataSet[:,j]) - minJ)

        centroids[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1))

    return centroids



def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):

    m = np.shape(dataSet)[0]

    clusterAssment = np.mat(np.zeros((m,2)))  #存储当前样迭代情况下，样本点属于哪一个K（簇）


    centroids = createCent(dataSet, k)

    clusterChanged = True   # 设置停止迭代条件

    while clusterChanged:

        clusterChanged = False

        for i in range(m):   # 对于每个数据点，将其分配给最近的质心。

            minDist = np.inf

            minIndex = -1

            for j in range(k):   # 让每一个样本点和每一个簇之间计算距离

                distJI = distMeas(centroids[j,:],dataSet[i,:])   # 计算两个点之间的欧式距离

                if distJI < minDist:

                    minDist = distJI
                    minIndex = j

            if clusterAssment[i,0] != minIndex:

                clusterChanged = True

            clusterAssment[i,:] = minIndex,minDist**2

        # print(centroids)

        for cent in range(k):     # 重新计算质心

            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]  # 获取每一个簇中的所有样本点

            centroids[cent,:] = np.mean(ptsInClust, axis=0) # 将质心指定为其中所有样本点距离的平均值
        print(ptsInClust.shape)

    return centroids, clusterAssment



if __name__ == '__main__':
    dataMat = np.mat(loadDataSet('./data/testSet.txt'))
    k=4
    a,b=kMeans(dataMat, k, distMeas=distEclud, createCent=randCent)
    print('每个样本点的聚类分析', b)
    print('质心位置',a)

    print('======对照试验=====')

    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=4)
    model.fit(dataMat)
    center=model.cluster_centers_
    print("中心点坐标:",center)

    order=pairwise_distances_argmin(X=a, Y=center)
    for i in range(k):
        print('中心点%s预测误差%s'%(i,a[i]-center[order[i]]))
