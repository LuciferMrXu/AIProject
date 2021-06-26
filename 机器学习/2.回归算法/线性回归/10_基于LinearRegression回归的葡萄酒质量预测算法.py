# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/4
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

mpl.rcParams['font.sans-serif'] = [u'simHei']

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = '../datas/winequality-red.csv'
df = pd.read_csv(path, sep=';')

# 2. 数据清洗
# NOTE: 不需要做数据处理

# # 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
X = df.drop('quality', axis=1)
Y = df['quality']
print("实际的等级:{}".format(np.unique(Y)))

# 4. 数据分割
# train_size: 给定划分之后的训练数据的占比是多少，默认0.75
# random_state：给定在数据划分过程中，使用到的随机数种子，默认为None，使用当前的时间戳；给定非None的值，可以保证多次运行的结果是一致的。
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 5. 特征工程的操作
# NOTE: 不做特征工程


# 6. 模型对象的构建
algo = LinearRegression(fit_intercept=True)

# 7. 模型的训练
algo.fit(x_train, y_train)

# 8. 模型效果评估
train_predict = algo.predict(x_train)
print(y_train[:20].ravel())
print(train_predict[:20])
train_predict = np.around(train_predict, 0).astype(np.int32)
print(train_predict[:20])
test_predict = np.around(algo.predict(x_test), 0).astype(np.int32)
# a. 查看模型训练好的相关参数
print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(algo.coef_))
print("截距项值:{}".format(algo.intercept_))
# b. 直接通过评估相关的API查看效果
train_score = np.mean(np.equal(y_train, train_predict))
test_score = np.mean(np.equal(y_test, test_predict))
print("模型在训练数据上的效果(准确率)：{}".format(train_score))
print("模型在测试数据上的效果(准确率)：{}".format(test_score))
