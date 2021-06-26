# -- encoding:utf-8 --
"""
    1. 加载恢复模型
    2. 使用恢复好的模型对需要预测的数据做一个预测即可
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


from sklearn.externals import joblib   # 保存加载模型

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
path = 'datas/household_power_consumption_1000.txt'
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

# 4. 加载模型
algo = joblib.load('G:/MongoDB/model/linear')

# 5. 使用加载的模型对数据做预测
y_hat = algo.predict(X)

# 模型效果输出
print("模型效果：{}".format(algo.score(X, Y)))

# 画图看一下效果如何
t = np.arange(len(Y))
plt.figure(facecolor='w')
plt.plot(t, Y, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='lower right')
plt.title('线性回归')
plt.show()