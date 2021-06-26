# -- encoding:utf-8 --
"""
Create by ibf on 2018/9/9
"""

import pandas as pd
import numpy as np
import matplotlib as mpl

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 一、加载数据
names = ['A', 'B', 'C', 'D', 'label']
path = '../datas/iris.data'
df = pd.read_csv(path, sep=',', header=None, names=names)

# 二、数据的清洗
df.replace('?', np.nan, inplace=True)
df.dropna(axis=0, how='any', inplace=True)

# 三、基于业务提取最原始的特征属性X和目标属性Y
X = df[names[:-1]]
Y = df[names[-1]]
# LabelEncoder：可以将输入的Y值数据转换为从0开始的数值型数据，认为传入的数据为类别数据
le = LabelEncoder()
# 这里的模型训练主要是为了找出传入的name和输出的index之间的映射关系
Y = le.fit_transform(Y)

# 四、数据的划分(将数据划分为训练集和测试集)
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75,test_size=0.25,random_state=16)

# 六、算法模型的选择/算法模型对象的构建
algo = SVC(C=0.1, kernel='rbf', probability=True,gamma='scale')

# 七、算法模型的训练
algo.fit(x_train, y_train)

# 八、模型效果评估
y_hat = algo.predict(x_test)

print("*" * 50)
print("训练集上的准确率:{}".format(algo.score(x_train, y_train)))
print("测试集上的准确率:{}".format(algo.score(x_test, y_test)))
print("训练集上的F1值:{}".format(f1_score(y_train, algo.predict(x_train), average='macro')))
print("测试集上的F1值:{}".format(f1_score(y_test, algo.predict(x_test), average='macro')))
print("训练集上的召回率:{}".format(recall_score(y_train, algo.predict(x_train), average='macro')))
print("测试集上的召回率:{}".format(recall_score(y_test, algo.predict(x_test), average='macro')))

# 九、输出分类中特有的一些API
print("=" * 100)
y_predict = algo.predict(x_test)
print("预测值:\n{}".format(y_predict))
print("预测的实际类别:\n{}".format(le.inverse_transform(y_predict)))
print("=" * 100)
# AttributeError: predict_proba is not available when  probability=False
print("属于各个类别的概率值:\n{}".format(algo.predict_proba(x_test)))
print("=" * 100)


# 打印模型参数
print("支持向量样本对应的下标:{}".format(algo.support_))
# iloc：获取DataFrame的对应行(根据行索引值)
# 获取第37行的数据(为了看一下支持向量对不)
print(x_train.iloc[10])
print("支持向量样本:{}".format(algo.support_vectors_))
print("每个类别的支持向量数量:{}".format(algo.n_support_))

print("决策函数返回值类型:{}".format(type(algo.decision_function(x_test.iloc[:2]))))
print("决策函数返回值:{}".format(algo.decision_function(x_test.iloc[:2])))