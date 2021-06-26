# -- encoding:utf-8 --
"""
Create by ibf on 2018/9/15
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 1. 加载数据
data = datasets.load_digits()
# 总样本数目是: 1797张0~9的图像；每张图像的长宽为:8 * 8， 使用64个像素点来表示这个图像信息；
print(data.images.shape)
print(data.data.shape)
print(data.target.shape)

# i = 100
# plt.imshow(data.images[i], cmap=plt.cm.gray_r)
# plt.title(data.target[i])
# plt.show()

# 2. 数据的划分(前1/2作为训练数据，后1/2作为测试数据)
n_samples = data.images.shape[0]
X = data.data
Y = data.target
x_train, x_test = X[:n_samples // 2], X[n_samples // 2:]
y_train, y_test = Y[:n_samples // 2], Y[n_samples // 2:]

# 3. 做一个SVM分类器
algo = SVC(gamma=0.001)
# algo = KNeighborsClassifier()

# 4. 模型构建
algo.fit(x_train, y_train)

# 5. 模型效果输出
print("训练数据上的效果:{}".format(algo.score(x_train, y_train)))
print("测试数据上的效果:{}".format(algo.score(x_test, y_test)))
