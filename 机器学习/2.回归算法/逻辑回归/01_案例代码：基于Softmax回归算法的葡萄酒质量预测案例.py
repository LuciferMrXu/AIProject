# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/11
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix

mpl.rcParams['font.sans-serif'] = [u'simHei']

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = '../datas/winequality-red.csv'
df = pd.read_csv(path, sep=';')
"""
如果直接用Softmax对葡萄酒的数据做一个预测，存在效果不好的问题，主要原因是：数据存在数据不平衡的情况，有部分类别的样本数据量很少，这样在模型训练的时候，学习不到该类别的特征信息。
解决方案：
  -1. 通过参数class_weight给定各个类别的权重，让模型训练的时候对于小众样本的数据具有更高的权重信息
  -2. 可以考虑将小众样本的数据作为一个单独的类别来做模型训练
  -3. 建议考虑一下决策树或者其他分类算法
Logistic回归算法更适合来处理二分类的问题。
"""

# df = df.loc[df.quality.isin([3, 4, 7, 8])]
# df[df.quality == 3] = 4
# df[df.quality == 7] = 4
# df[df.quality == 8] = 4
# df = df.loc[~ df.quality.isin([3, 4, 7, 8])]
# df = df.loc[df.quality.isin([3, 4, 7, 8])]
# df = df.loc[df.quality.isin([3, 4, 8])]
# df = df.loc[df.quality.isin([3, 4])]


# 2. 数据清洗
# NOTE: 不需要做数据处理

# # 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
X = df.drop('quality', axis=1)
Y = df['quality']
print("实际的等级:{}".format(np.unique(Y)))
print(Y.value_counts())

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
"""
class_weight: 只在分类API中存在，作用是给定各个类别的权重因子，权重越大的样本在模型训练的时候会着重考虑;eg;class_weight={3: 100, 4: 10, 5: 1, 6: 1, 7: 5, 8: 50}
multi_class: 给定多分类的求解方式，可选参数：ovr和multinomial，multinomial相当于softmax； 默认是ovr
"""
algo = LogisticRegression(multi_class='multinomial',solver='lbfgs')

# 7. 模型的训练
algo.fit(x_train, y_train)

# 8. 模型效果评估
train_predict = algo.predict(x_train)
test_predict = algo.predict(x_test)
print("测试集上的效果(准确率):{}".format(algo.score(x_test, y_test)))
print("训练集上的效果(准确率):{}".format(algo.score(x_train, y_train)))
print("测试集上的效果(准确率):{}".format(accuracy_score(y_test, test_predict)))
print("训练集上的效果(准确率):{}".format(accuracy_score(y_train, train_predict)))
print("测试集上的效果(混淆矩阵):\n{}".format(confusion_matrix(y_test, test_predict)))
print("训练集上的效果(混淆矩阵):\n{}".format(confusion_matrix(y_train, train_predict)))


# 9. 调整阈值
print("参数值theta:\n{}".format(algo.coef_))
print("参数值截距项:\n{}".format(algo.intercept_))
print(test_predict)
print("decision_function API返回值:\n{}".format(algo.decision_function(x_test)))
print("sigmoid函数返回的概率值:\n{}".format(algo.predict_proba(x_test)))
print("sigmoid函数返回的经过对数转换之后的概率值:\n{}".format(algo.predict_log_proba(x_test)))