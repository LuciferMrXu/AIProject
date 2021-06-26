# -- encoding:utf-8 --
"""
Create by ibf on 2018/9/8
"""
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

# make_blobs：产生一个服从给定的均值和标准差的高斯分布的数据，最终返回数据样本所组成的x特征矩阵，以及样本所对应的类别(当前数据是属于哪个均值哪个标准差的数据分布)
x, y = make_blobs(n_samples=5, n_features=2, centers=3)
print(x)
print(y)

# 2. 数据的可视化
# 参数c: 给定坐标轴上对应点的类别
plt.scatter(x[:, 0], x[:, 1], c=y, s=30)
plt.show()
