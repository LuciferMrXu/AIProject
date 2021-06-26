# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/11
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

mpl.rcParams['font.sans-serif'] = [u'simHei']

# np.random.seed(0)

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = '../datas/iris.data'
names = ['A', 'B', 'C', 'D', 'cla']
df = pd.read_csv(path, header=None, names=names)

# 2. 数据清洗

# # 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
X = df[names[0:-1]]
Y = df[names[-1]]
# print(Y)
label_encoder = LabelEncoder()
label_encoder.fit(Y)
Y = label_encoder.transform(Y)
# 这里得到的序号其实就是classes_这个集合中对应数据的下标
# print(label_encoder.classes_)
# true_label = label_encoder.inverse_transform([0, 1, 2, 0])
# print(true_label)
# print(Y)

# 4. 数据分割
# train_size: 给定划分之后的训练数据的占比是多少，默认0.75
# random_state：给定在数据划分过程中，使用到的随机数种子，默认为None，使用当前的时间戳；给定非None的值，可以保证多次运行的结果是一致的。
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8,test_size=0.2, random_state=28)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 5. 特征工程的操作
# NOTE: 不做特征工程

# 6. 模型对象的构建
"""
# 随机森林中，子模型决策树的数目。默认为10
n_estimators=10,
# 同决策树算法
criterion="gini",
# 同决策树算法
max_depth=None,
# 同决策树算法
min_samples_split=2,
min_samples_leaf=1,
min_weight_fraction_leaf=0.,
# 同决策树算法
max_features="auto",
max_leaf_nodes=None,
min_impurity_split=1e-7,
# 给定是否采用有放回的方式产生子数据集，默认为True表示采用。
bootstrap=True,
oob_score=False,
n_jobs=1,
random_state=None,
verbose=0,
warm_start=False,
class_weight=None
"""
algo = RandomTreesEmbedding(n_estimators=10, max_depth=2, sparse_output=False)

# 7. 模型的训练
algo.fit(x_train)

# 8. 直接获取扩展结果
x_train2 = algo.transform(x_train)
x_test2 = algo.transform(x_test)
print("扩展前大小:{}， 扩展后大小:{}".format(x_train.shape, x_train2.shape))
print("扩展前大小:{}， 扩展后大小:{}".format(x_test.shape, x_test2.shape))

# 9.输出随机森林中各个特征属性的重要性权重系数
print("所有特征列表:{}".format(names[0:-1]))
print("对应特征属性的重要性权重:\n{}".format(algo.feature_importances_))




# 10. 其他特殊的API
print("子模型列表:\n{}".format(algo.estimators_))

from sklearn import tree
import pydotplus

k = 0
for algo1 in algo.estimators_:
    dot_data = tree.export_graphviz(decision_tree=algo1, out_file=None,
                                    feature_names=['A', 'B', 'C', 'D'],
                                    class_names=['1', '2', '3'],
                                    filled=True, rounded=True,
                                    special_characters=True
                                    )

    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('trte_{}.png'.format(k))
    k += 1
    if k > 3:
        break

# 做一个维度扩展
print("*" * 100)
x_test2 = x_test.iloc[:2, :]
print(x_test2)
# apply方法返回的是叶子节点下标
print(algo.apply(x_test2))
# transform转换数据（其实就是apply方法+哑编码）
print(algo.transform(x_test2))
