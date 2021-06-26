# -- encoding:utf-8 --
"""
Create by ibf on 2018/9/16
"""
# http://scipy.github.io/devdocs/generated/scipy.stats.multivariate_normal.html#scipy.stats.multivariate_normal

import numpy as np
from sklearn.mixture import GaussianMixture

# 1. 产生模拟数据
np.random.seed(28)
N1 = 400
N2 = 100
# 类别1的数据
mean1 = (0, 0, 0)
cov1 = np.diag((1, 2, 3))
data1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=N1)
# 类别2的数据
mean2 = (5, 5, 5)
cov2 = np.array([
    [3, 1, 0],
    [1, 1, 0],
    [0, 0, 5]
])
data2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=N2)
# 合并两个数据
data = np.vstack((data1, data2))
# 构建对应的

# 2. 模型构建
num_iter = 1000
m, d = data.shape
print("样本数目:{}，每个样本的维度:{}".format(m, d))

# 构建模型以及训练
"""
n_components: 当做聚类的类别数目
covariance_type='full': 方差、协方差矩阵的初始化方式，可选值: full、tied、diag、spherical
"""
algo = GaussianMixture(n_components=2, covariance_type='full')
algo.fit(data)

print("构建好的pi:\n{}".format(algo.weights_))
print("构建好的均值:\n{}".format(algo.means_))
print("构建好的协方差:\n{}".format(algo.covariances_))

# 3. 基于训练好的模型参数对数据做一个预测(数据的聚类划分)
x_test = np.array([
    [0.0, 0.0, 0.0],
    [2.5, 1.5, 1.5],
    [5.0, 5.0, 5.0],
    [6.0, 5.0, 7.0]
])
y_hat = algo.predict(x_test)
print("预测的类别为:{}".format(y_hat))
print("预测为各个类别的概率为:\n{}".format(algo.predict_proba(x_test)))
