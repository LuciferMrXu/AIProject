# -- encoding:utf-8 --
"""
只要是机器学习领域，编程的流程基本和该文件中的内容一致
Create by ibf on 2018/11/10
"""

import time
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline

mpl.rcParams['font.sans-serif'] = [u'simHei']

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = '../datas/household_power_consumption_1000_2.txt'
df = pd.read_csv(path, sep=';')

# 2. 数据清洗
df.replace('?', np.nan, inplace=True)
df = df.dropna(axis=0, how='any')


# 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
def date_format(dt):
    date_str = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return [date_str.tm_year, date_str.tm_mon, date_str.tm_mday, date_str.tm_hour, date_str.tm_min, date_str.tm_sec]


X = df.iloc[:, 0:2]
X = X.apply(lambda row: pd.Series(date_format(row)), axis=1)
Y = df.iloc[:, 4].astype(np.float32)

# 4. 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=0)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 5. 管道流对象构建
"""
RidgeCV算法的参数：
alphas=(0.1, 1.0, 10.0), 给定Ridge算法中，alpha参数的选择可能列表
fit_intercept=True, 给定是否训练截距项 
normalize=False, 是否对象数据做一个归一化处理
scoring=None, 给定做交叉验证的时候选择最优模型的方式，默认使用Ridge算法的score方法来作为最优模型的选择
cv=None, 给定做几折交叉验证
"""
algo = Pipeline(steps=[
    ('poly', PolynomialFeatures(degree=2)),  # 指定第一步做什么操作
    ('algo', RidgeCV(alphas=[0.1, 0.01, 0.001, 1.0, 10.0], cv=5))  # 指定最后一步做什么操作，最后一步一般为模型对象
])

# 6. 模型的训练
algo.fit(x_train, y_train)

# 7. 模型效果评估
print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(algo.get_params()['algo'].coef_))
print("截距项值:{}".format(algo.steps[-1][1].intercept_))
print("Ridge算法中的最优模型参数:{}".format(algo.steps[-1][-1].alpha_))
# b. 直接通过评估相关的API查看效果
print("模型在训练数据上的效果(R2)：{}".format(algo.score(x_train, y_train)))
# 在测试的时候对特征属性数据必须做和训练数据完全一样的操作
print("模型在测试数据上的效果(R2)：{}".format(algo.score(x_test, y_test)))
