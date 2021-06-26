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
from sklearn.linear_model import ElasticNet
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
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 5. 管道流对象构建
"""
ElasticNet算法的参数：
alpha=1.0, :ppt上的lambda，也就是给定惩罚项/正则项系数，该值越大，那么要求最终模型的参数越小
l1_ratio=0.5: ppt上的p，也就是在惩罚项中，L1正则的占比是多少
fit_intercept=True, 给定是否训练截距项 
normalize=False, 是否对象数据做一个归一化处理
precompute=False：是否做预训练，该参数不要改动
copy_X=True, 是否copy数据训练
max_iter=1000, 训练过程中的最大迭代次数
tol=1e-3, 训练的收敛值
selection='cyclic': 给定模型的训练过程，cyclic表示循环训练，可选值：random
random_state=None : 给定算法中用到的随机数种子
"""
algo = Pipeline(steps=[
    ('poly', PolynomialFeatures(degree=4)),  # 指定第一步做什么操作
    ('algo', ElasticNet(alpha=1.0, l1_ratio=0.1))  # 指定最后一步做什么操作，最后一步一般为模型对象
])

# 6. 模型的训练
algo.fit(x_train, y_train)

# 7. 模型效果评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(algo.get_params()['algo'].coef_))
print("截距项值:{}".format(algo.steps[-1][1].intercept_))

pred_train = algo.predict(x_train)
pred_test = algo.predict(x_test)
# b. 直接通过评估相关的API查看效果
print("模型在训练数据上的效果(R2)：{}".format(r2_score(pred_train, y_train)))
print("模型在测试数据上的效果(R2)：{}".format(r2_score(pred_test, y_test)))
print("模型在训练数据上的效果(MSE)：{}".format(mean_squared_error(pred_train, y_train)))
print("模型在测试数据上的效果(MSE)：{}".format(mean_squared_error(pred_test, y_test)))
print("模型在训练数据上的效果(MAE)：{}".format(mean_absolute_error(pred_train, y_train)))
print("模型在测试数据上的效果(MAE)：{}".format(mean_absolute_error(pred_test, y_test)))
print("模型在训练数据上的效果(RMSE)：{}".format(np.sqrt(mean_squared_error(pred_train, y_train))))
print("模型在测试数据上的效果(RMSE)：{}".format(np.sqrt(mean_squared_error(pred_test, y_test))))
