# -- encoding:utf-8 --
"""
Create by ibf on 2018/9/8
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

# 1. 产生模拟数据
N = 1000
n_centers = 4
X, Y = make_blobs(n_samples=N, n_features=2, centers=n_centers, random_state=14)

# 3. 模型构建
parameters = {
    'n_clusters': [2, 3, 4, 5, 6],
    'random_state': [0, 14, 28]
}
model = KMeans(n_clusters=n_centers)
algo = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
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
# 这里输出的0 1 2 3只能表示某个类别, 这个类别到底是啥不知道，其实这个数字是algo.cluster_centers_对应的中心点下标
print("预测值为:{}".format(algo.predict(x_test)))

print("最优的模型参数:")
print(algo.best_params_)
print("中心点坐标:")
print(algo.best_estimator_.cluster_centers_)
print("目标函数的损失值:(所有样本到对应簇中心点的距离平方和)")
print(algo.best_estimator_.inertia_)
print(algo.best_estimator_.inertia_ / N)

# 2. 数据的可视化
# 参数c: 给定坐标轴上对应点的类别
plt.scatter(X[:, 0], X[:, 1], c=Y, s=30)
plt.show()
