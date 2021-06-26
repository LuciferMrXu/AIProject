# -- encoding:utf-8 --
"""
功能：降低代码的编程量
"""

import time
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

mpl.rcParams['font.sans-serif'] = [u'simHei']

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = '../datas/household_power_consumption_1000_2.txt'
df = pd.read_csv(path, sep=';')

# 2. 数据的清洗
# inplace: 当设置为True的时候，表示对原始的DataFrame做修改；默认为False
df.replace('?', np.nan, inplace=True)
# DataFrame: axis=0表示对行做处理，axis=1表示对列做处理
# how: 可选参数any和all，any表示只要有任意一个为nan，那么进行数据删除；如果为all，表示所有值为nan的时候数据删除
# 功能：只要有任意一个样本中的任意数据特征属性为nan的形式，就就将该样本删除。
df = df.dropna(axis=0, how='any')
# print(df.info())


# 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
def date_format(dt):
    date_str = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return [date_str.tm_year, date_str.tm_mon, date_str.tm_mday, date_str.tm_hour, date_str.tm_min, date_str.tm_sec]


X = df.iloc[:, 0:2]
X = X.apply(lambda row: pd.Series(date_format(row)), axis=1)
Y = df.iloc[:, 4].astype(np.float64)

# 4. 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 5. 管道流对象构建

model = Pipeline(steps=[
    ('ss',StandardScaler()),   # 指定第一步做什么操作
    ('poly', PolynomialFeatures(degree=4)),
    ('algo', LinearRegression(fit_intercept=True))  # 指定最后一步做什么操作，最后一步一般为模型对象
])


# 设置参数的时候根据设置的步骤名称以及对应步骤对象的属性值来设置参数
model.set_params(algo__fit_intercept=False)  # 通过‘名称__属性’设置参数
print(model.get_params()['algo'])  # 根据你给定的名称来获取


# 6. 模型的训练
model.fit(x_train, y_train)

# 7. 模型效果评估
# 可选，主要目的就是评估一下模型的效果
# 主要就是根据算法类型，分别选择精确率、准确率、F1、RMSE、MSE、R2等指标进行校验
print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(model.get_params()['algo'].coef_))
print("截距项值:{}".format(model.steps[-1][1].intercept_))
# b. 直接通过评估相关的API查看效果
print("模型在训练数据上的效果(R2)：{}".format(model.score(x_train, y_train)))
# 在测试的时候对特征属性数据必须做和训练数据完全一样的操作
print("模型在测试数据上的效果(R2)：{}".format(model.score(x_test, y_test)))

# 8. 管道流输出保存
from sklearn.externals import joblib

"""
两种保存方式：
1. 直接保存模型对象
2. 保存模型的预测结果到数据库或者其它数据存储的位置
"""
# a. 直接保存模型对象
joblib.dump(model, 'G:/MongoDB/model/pipline')

# b. 保存预测结果
y_hat = model.predict(x_test) # 获取预测结果
print(y_hat)


# 9. 画图看一下效果如何
t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='lower right')
plt.title('线性回归')
plt.show()