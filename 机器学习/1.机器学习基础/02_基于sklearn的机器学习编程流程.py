# -- encoding:utf-8 --
"""
只要是机器学习领域，编程的流程基本和该文件中的内容一致
Create by ibf on 2018/11/4
"""

# 1. 加载数据(数据一般存在于磁盘或者数据库)

# 2. 数据清洗

# 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y

# 4. 数据分割

# 5. 特征工程的操作

# 6. 模型对象的构建

# 7. 模型的训练

# 8. 模型效果评估

# 9. 模型保存\模型持久化
"""
方式一：直接保存预测结果
方式二：将模型持久化为磁盘文件
方式三：将模型参数保存数据库
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split   # 训练集测试集
from sklearn.linear_model import LinearRegression     # 线性回归模型
from sklearn.externals import joblib       # 模型保存
from sklearn.preprocessing import StandardScaler     # 标准化

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
path = './datas/household_power_consumption_201.txt'
df = pd.read_csv(filepath_or_buffer=path, sep=';')
# print(df.info())


# 2. 数据的清洗
# inplace: 当设置为True的时候，表示对原始的DataFrame做修改；默认为False
df.replace('?', np.nan, inplace=True)
# DataFrame: axis=0表示对行做处理，axis=1表示对列做处理
# how: 可选参数any和all，any表示只要有任意一个为nan，那么进行数据删除；如果为all，表示所有值为nan的时候数据删除
# 功能：只要有任意一个样本中的任意数据特征属性为nan的形式，就就将该样本删除。
df = df.dropna(axis=0, how='any')
# print(df.info())


# 3. 模型所需要的特征属性获取(构建特征属性矩阵X和目标属性Y)
X = df.iloc[:, 2:4]
Y = df.iloc[:, 5]


# 4. 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7,test_size=0.3, random_state=16)


# 5. 特征工程(操作可选)
# 主要就是：哑编码（对没有大小关系的离散数据做onehot）、连续数据离散化、数据的转换、标准化、归一化、降维.....
ss = StandardScaler()     # 数据标准化——消除数据量纲，变成标准正态分布(一般只对连续型数据做，离散数据除非方差特别大，否则不做)
# x_train = ss.fit_transform(x_train)          # 训练并转换
x_train = ss.fit_transform(X)          # 直接用全部数据做训练集效果更好
x_test = ss.transform(x_test)          # 直接使用在模型构建数据上进行一个数据标准化操作


# 6. 算法/模型对象构建
algo = LinearRegression()     # 线性回归模型


# 7. 算法模型训练
algo.fit(x_train, Y)


# 8. 模型效果评估
# 可选，主要目的就是评估一下模型的效果
# 主要就是根据算法类型，分别选择精确率、准确率、F1、RMSE、MSE、R2等指标进行校验
print("模型效果：{}".format(algo.score(x_train, Y)))


# 9. 模型的保存
"""
两种保存方式：
1. 直接保存模型对象
2. 保存模型的预测结果到数据库或者其它数据存储的位置
"""
# a. 直接保存模型对象
joblib.dump(algo, 'G:/MongoDB/model/linear')

# b. 保存预测结果
# 获取预测结果,然后导入数据库
y_hat = algo.predict(x_test)
print(y_hat)


# 10. 可视化绘图
t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='lower right')
plt.title('线性回归')
plt.show()
