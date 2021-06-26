# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/11
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import label_binarize

mpl.rcParams['font.sans-serif'] = [u'simHei']

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = './datas/iris.data'
names = ['A', 'B', 'C', 'D', 'cla']
df = pd.read_csv(path, header=None, names=names)
# print(df.head())
# print(df.info())



# 2. 数据清洗
# NOTE: 不需要做数据处理
def parse_record(row):
    result = []
    r = zip(names, row)
    for name, value in r:
        if name == 'cla':
            if value == 'Iris-setosa':
                result.append(1)
            elif value == 'Iris-versicolor':
                result.append(2)
            elif value == 'Iris-virginica':
                result.append(3)
            else:
                result.append(0)    # 可利通过过滤0来清洗数据
        else:
            result.append(value)
    return result


df = df.apply(lambda row: pd.Series(parse_record(row), index=names), axis=1)   # axis=1按行处理
df.cla = df.cla.astype(np.int32)
# print(df.head())
# print(df.cla)



# 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
X = df[names[0:-1]]
Y = df[names[-1]]


# 4. 数据分割
# train_size: 给定划分之后的训练数据的占比是多少，默认0.75
# random_state：给定在数据划分过程中，使用到的随机数种子，默认为None，使用当前的时间戳；给定非None的值，可以保证多次运行的结果是一致的。
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8,test_size=0.2, random_state=28)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 5. 特征工程的操作
# NOTE: 不做特征工程

# 6. KNN模型对象的构建
"""
n_neighbors=5, 使用多少个邻居，也就是KNN中的K值
weights='uniform', 各个样本的权重系数，可选值: uniform、distance；uniform表示所有样本等权重，distance表示样本具有不一样的权重是和距离成反比
algorithm='auto', 模型的求解方式，默认为auto；可选值：auto、brute、kd_tree、ball_tree
leaf_size=30, 当模型求解方式为kd_tree或者ball_tree的时候，树中最多允许的叶子数目
p=2, 表示在minkowski距离中，变成为欧几里得距离公式
metric='minkowski',  给定距离公式计算方式，可选参数：https://scikit-learn.org/0.18/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
metric_params=None, 在距离公式的计算中，参数列表
"""
algo = KNeighborsClassifier(n_neighbors=5)



# 7. 模型的训练
algo.fit(x_train, y_train)



# 8. 模型效果评估
train_predict = algo.predict(x_train)
test_predict = algo.predict(x_test)
print("测试集上的效果(准确率):{}".format(algo.score(x_test, y_test)))
print("训练集上的效果(准确率):{}".format(algo.score(x_train, y_train)))
print("测试集上的效果(分类评估报告):\n{}".format(classification_report(y_test, test_predict)))
print("训练集上的效果(分类评估报告):\n{}".format(classification_report(y_train, train_predict)))


flag = False
if flag:
    # 将三个类别的数据合并一起来计算AUC的值
    y_true = label_binarize(y_test, classes=(1, 2, 3))
    y_score = algo.predict_proba(x_test)
    fpr, tpr, threads = metrics.roc_curve(y_true.ravel(), y_score.ravel())
    auc = metrics.auc(fpr, tpr)
    print(auc)
else:
    # 对于三个类别，分别计算auc的值
    y_true = label_binarize(y_test, classes=(1, 2, 3))
    y_score = algo.predict_proba(x_test)

    fpr, tpr, threads = metrics.roc_curve(y_true[:, 0], y_score[:, 0])
    auc = metrics.auc(fpr, tpr)
    print("对于类别1的单独计算AUC的值:{}".format(auc))
    fpr, tpr, threads = metrics.roc_curve(y_true[:, 1], y_score[:, 1])
    auc = metrics.auc(fpr, tpr)
    print("对于类别2的单独计算AUC的值:{}".format(auc))
    fpr, tpr, threads = metrics.roc_curve(y_true[:, 2], y_score[:, 2])
    auc = metrics.auc(fpr, tpr)
    print("对于类别3的单独计算AUC的值:{}".format(auc))

# 9. 其它
print(y_test.ravel())
print(test_predict)
print("softmax/sigmoid函数返回的概率值:\n{}".format(algo.predict_proba(x_test)))

test1 = x_test.iloc[:10, :]
print(test1)
"""
kneighbors_graph：在训练数据中获取和当前传入数据最相似的n_neighbors的3个样本信息，mode如果为distance，返回的是距离；如果值为：connectivity，返回n_neighbors个连通的样本点；一般情况下，修改为distance
"""
graph1 = algo.kneighbors_graph(test1, n_neighbors=3, mode='connectivity')
print(graph1)

print("*" * 100)
"""
直接返回给定x在训练数据中最相似的n_neighbors个样本，return_distance表示是否返回距离，设置为True表示返回距离；设置为False表示返回样本的索引值
"""
neighbors = algo.kneighbors(test1, n_neighbors=5, return_distance=False)
print(neighbors)
