# -- encoding:utf-8 --
"""
Create by ibf on 2018/9/9
"""

import pandas as pd
import numpy as np
import matplotlib as mpl

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import recall_score,f1_score

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
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8,test_size=0.2, random_state=28)

# 六、算法模型的选择/算法模型对象的构建
model = SVC(probability=True)
parameters = {
    'C': [0.01, 0.1, 0.5, 1.0],
    'kernel': ['rbf', 'poly'],
    'gamma': ['auto', 0.001, 0.01, 0.1, 1.0],
    'degree': [2, 3]
}
algo = GridSearchCV(estimator=model, param_grid=parameters, cv=3)

# 七、算法模型的训练
algo.fit(x_train, y_train)
# 获取实际最优模型
print("最优模型:{}".format(algo.best_estimator_))
print("最优模型对应的参数:{}".format(algo.best_params_))

# 八、模型效果评估
y_hat = algo.predict(x_test)
# 看方法的注释的方法：按ctrl键，同时鼠标在方法上左键点击
print("训练集上的准确率:{}".format(algo.best_estimator_.score(x_train, y_train)))
print("测试集上的准确率:{}".format(algo.best_estimator_.score(x_test, y_test)))
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
