#_*_ coding:utf-8_*_
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 加载数据
path = './DATA/regress/dq_unisex_names.csv'
df = pd.read_csv(filepath_or_buffer=path,header=None,sep=',')
df.columns=['name','salary']



# 模型所需要的特征属性获取(构建特征属性矩阵X和目标属性Y)
value=[]
name = df.iloc[:, 0]
for i, element in enumerate(name):
    print(i, element)
    value.append(i)

X = np.array(value).reshape(-1,1)
Y = df['salary']



poly = PolynomialFeatures(degree=5)
X_train = poly.fit_transform(X)



algo1 = LinearRegression(fit_intercept=True)#创建回归器
algo1.fit(X_train, Y)#训练数据

print("PF模型的效果(R2)：{}".format(algo1.score(X_train, Y)))

# 模型对象的构建
algo2 = GradientBoostingRegressor(n_estimators=10, learning_rate=1.0, max_depth=3)

# 模型的训练
algo2.fit(X, Y)

# 模型效果评估
# 直接通过评估相关的API查看效果
print("GBDT模型的效果(R2)：{}".format(algo2.score(X, Y)))


y_hat1=algo1.predict(X_train)
y_hat2=algo2.predict(X)
x_hat = np.linspace(X.min(), X.max(), num=len(X)) ## 产生模拟数据
x_hat.shape = -1,1


plt.figure(facecolor='w')

plt.subplot(121)
plt.scatter(X, Y, color='r',label='实际值')
plt.plot(x_hat, y_hat1, color='b', linewidth=3,label='回归曲线')
plt.legend(loc = 'best')
plt.title("多项式回归", fontsize=20)

plt.subplot(122)
plt.scatter(X, Y, color='r',label='实际值')
plt.plot(x_hat, y_hat2, color='b', linewidth=3,label='回归曲线')
plt.legend(loc = 'best')
plt.title("GBDT", fontsize=20)

plt.show()