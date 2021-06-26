# -- encoding:utf-8 --
"""
手写数字图片识别
Create by ibf on 2018/7/15
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# 1. 加载数据
data = datasets.load_digits()
print(type(data))
print(data)
# 总的图像数目是1797个图像，每个图像是一个8*8的灰度图
print(data.images.shape)
print(data.data.shape)
print(data.target.shape)  # target是图像是实际数字

# 2. 获取X和Y
n_samples = data.images.shape[0]
X = data.images
X = X.reshape((X.shape[0], -1))
Y = data.target
print("特征矩阵信息:样本数目{}, 每个样本的特征数目:{}".format(X.shape[0], X.shape[1]))
print("待预测的目标属性数量:{}".format(Y.shape))

# 划分训练集和测试集
x_train, x_test = X[:int(n_samples / 2)], X[int(n_samples / 2):]
y_train, y_test = Y[:int(n_samples / 2)], Y[int(n_samples / 2):]

# 3. 构建分类模型
algo = svm.SVC(gamma='scale',C=100)
# algo = KNeighborsClassifier(n_neighbors=10)
algo.fit(x_train, y_train)

# 4. 输出一下模型效果
print("训练集上的准确率:{}".format(algo.score(x_train, y_train)))
print("测试集上的准确率:{}".format(algo.score(x_test, y_test)))

# 5. 画图看一下数据
index = 1024
image = X[index]
target = Y[index]
image = image.reshape((8, 8))
plt.imshow(image, cmap=plt.cm.gray_r)
plt.title(target)
plt.show()
