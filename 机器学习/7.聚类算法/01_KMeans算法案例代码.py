# -- encoding:utf-8 --
"""
聚类算法的主要作用：
 1. 作为数据的预处理技术；首先对数据做一个聚类，然后把中心点以及中心点附近的样本特征属性取值输出，人工查看一下这个特征属性的特征值，根据特征值给定每个簇所对应的实际类别y，然后基于给定标签的数据使用有监督的算法进行模型训练
 2. 在一些不关注样本簇具体类型的业务场景，并且从数据结构上来讲，确实是存在很大的数据特征的时候，用聚类算法。
Create by ibf on 2018/9/8
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# 1. 产生模拟数据
N = 1000         # 样本数
n_centers = 4    # 中心点数
X, Y = make_blobs(n_samples=N, n_features=2, centers=[(-5, 5), (0, 0), (5, -5), (5, 5)], random_state=14)
X, Y = make_blobs(n_samples=N, n_features=2, centers=n_centers, cluster_std=2.0, random_state=14)
X, Y = make_blobs(n_samples=N, n_features=2, centers=[(-15, 15), (0, 0), (15, -15), (13, 15)], cluster_std=2.0,
                  random_state=14)
X, Y = make_blobs(n_samples=N, n_features=2, centers=[(-15, 15), (0, 0), (15, -15), (13, 15)],
                  cluster_std=[1.0, 5.0, 3.0, 1.5], random_state=14)

X, Y = make_blobs(n_samples=N, n_features=2, centers=n_centers, random_state=14)

# 数据的可视化一下
plt.scatter(X[:, 0], X[:, 1], c=Y, s=30)
plt.show()


# 3. 模型构建
algo = KMeans(n_clusters=n_centers,n_init=10)  # n_init=10选十组初始节点进行聚类，最终选择score值最大的划分方式
algo.fit(X)

# 4. 对数据做预测
x_test = [
    [-4, 8],
    [-3, 7],
    [0, 5],
    [0, -5],
    [8, -8],
    [7, -9]
]

# print("中心点坐标:")
# print(algo.cluster_centers_)
# print("目标函数的损失值:(所有样本到对应簇中心点的距离平方和)")
# print(algo.inertia_)
# print(algo.inertia_ / N)
#
# # 这里输出的0 1 2 3只能表示某个类别, 这个类别到底是啥不知道，其实这个数字是algo.cluster_centers_对应的中心点下标
# print("预测值为:{}".format(algo.predict(x_test)))
# # score值=-loss值，因为score API的作用认为是：返回的值越大越好
# print("模型的Score评估值:{}".format(algo.score(X, Y)))


print(algo.labels_)
