# 特征筛选+数据分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from icecream import ic
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据加载
train = pd.read_csv(os.path.join(BASE_DIR,'zhengqi_train.txt'), sep='\t')
ic(train)

test = pd.read_csv(os.path.join(BASE_DIR,'zhengqi_train.txt'), sep='\t')
ic(test)

ic(train.info())
ic(train.describe())

# 可视化数据分布
plt.figure(figsize=(4, 6))
# 对v0特征进行箱线图垂直呈现
sns.boxplot(train['V0'], orient='v', width=0.5)


# 一共39列，可以画成5行*8列
cols = train.columns
# Ⅰ.通过箱线图找离群点
plt.figure(figsize=(40, 30))
i = 0
for col in cols:
    i = i + 1
    # 子图定位
    plt.subplot(5, 8, i)
    # 绘制箱线图
    sns.boxplot(train[col], orient='v', width=0.5)
    plt.ylabel(col, fontsize=18)


# 使用直方图对于所有的特征字段，查看训练集和测试集的分布
plt.figure(figsize=(40, 30))
i = 0
for col in cols[:-1]:
    i = i + 1
    # 设定画图的subplot
    g = plt.subplot(5, 8, i)
    sns.distplot(train[col],color='red',ax=g ,label='Train')
    sns.distplot(test[col],color="blue",ax=g ,label="Test")
    g.set_xlabel(col)
    g.set_ylabel("Frequency")



# Ⅱ.使用KDE图对于所有的特征字段，查看训练集和测试集的分布，对相差较大的分布做变换或者直接丢弃
plt.figure(figsize=(40, 30))
i = 0
for col in cols[:-1]:
    i = i + 1
    # 设定画图的subplot
    g = plt.subplot(5, 8, i)
    sns.kdeplot(train[col],color='red',ax=g ,label='Train',shade=True)
    sns.kdeplot(test[col],color="blue",ax=g ,label="Test",shade=True)
    g.set_xlabel(col)
    g.set_ylabel("Frequency")



# 找到训练集和测试集直接分布差距较大的特征，并丢弃
drop_cols = ["V5","V9","V11","V17","V22","V28"]
train2 = train.drop(drop_cols, axis=1)
test2 = test.drop(drop_cols, axis=1)

# 绘制特征V0与target之间的线性相关性
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
sns.regplot(x='V0', y='target', data=train2, line_kws={'color': 'red'})
plt.subplot(1, 2, 2)
sns.distplot(train2['V0'])


# 对于所有的特征 与 Target的线性回归关系
plt.figure(figsize=(40, 30))
i = 0

cols = list(set(cols[:-1]) ^ set(drop_cols))

for col in cols:
    i = i + 1
    # 设定画图的subplot
    g = plt.subplot(5, 8, i)
    # 绘制线性回归拟合图
    sns.regplot(x=col, y='target', data=train2, line_kws={'color': 'red'})


# 查看相关性系数
ic(train2.corr())

# 绘制相关性系数的热力图
plt.figure(figsize=(20, 16))
sns.heatmap(train2.corr(), annot=True)



# Ⅲ.根据相关性系数筛选出来重要的特征
threshold = 0.5
corrs = train2.corr()
features_filter = corrs.index[abs(corrs['target'])>threshold]
ic(features_filter)


# 对重要的特征做热力图
plt.figure(figsize=(16, 10))
sns.heatmap(train2[features_filter].corr(), annot=True)

# plt.show()

# 设置输入模型的Features
features_filter = features_filter.tolist()
features_filter.remove('target')
ic(features_filter)



# 数据归一化
ss = StandardScaler()
train2[features_filter] = ss.fit_transform(train2[features_filter])
test2[features_filter] = ss.transform(test2[features_filter])




"""
    使用Linear Regression
    筛选与target相关的系数 > 0.5
    没有做数据规范化，作为基线
    Score = 0.6290
"""
# 模型创建
model = LinearRegression()
model.fit(train2[features_filter], train2['target'])
y_pred = model.predict(test2[features_filter])
ic(y_pred)


y_pred = pd.DataFrame(y_pred)
y_pred.to_csv(os.path.join(BASE_DIR,'baseline.txt'), index=False, header=None)



ic(len(y_pred))
ic(test2)

