# -- encoding:utf-8 --
"""
用最小二乘解析式的方式求解
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

mpl.rcParams['font.sans-serif'] = [u'simHei']

# 1. 数据加载
path = './datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';')
# df.info()
# print(df.head(2))

# 2. 获取特征属性X和目标属性Y
X = df.iloc[:, 2:4]
# X['b'] = pd.Series(data=np.ones(shape=X.shape[0]))
Y = df.iloc[:, 5]
# print(X.head(5))
# print(Y[:5])

# 3. 划分训练数据集和测试数据集
# train_size: 给定划分之后的训练数据的占比是多少，默认0.75
# random_state：给定在数据划分过程中，使用到的随机数种子，默认为None，使用当前的时间戳；给定非None的值，可以保证多次运行的结果是一致的。
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 4. 模型构建
# a.使用numpy的API mat()将DataFrame转换成为矩阵的对象
x = np.mat(x_train)
y = np.mat(y_train).reshape(-1, 1)
# print(x.shape)
# print(y.shape)

# b. 直接解析式求解theta值
theta = (x.T * x).I * x.T * y   # 转化为矩阵后可以直接点乘
"""
[[ 4.2000866 ]
 [ 1.37131883]]
"""
print(theta)

# 5. 使用训练出来的模型参数theta对测试数据做一个预测
predict_y = np.mat(x_test) * theta

# 6. 可以考虑一下画图看一下效果
t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', label=u'真实值')
plt.plot(t, predict_y, 'b-', label=u'预测值')
plt.legend(loc='lower right')
plt.show()
