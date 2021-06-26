# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/11
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 5. 特征工程的操作
# NOTE: 不做特征工程

# 6. 模型对象的构建
"""
criterion="gini", -> 指定在计算最优划分特征的时候，特征属性的纯度衡量指标，可选值：gini和entropy
splitter="best", -> 指定在选择特征属性的时候的划分方式，可选值：best和random
max_depth=None, -> 剪枝参数，构建完成的决策树允许最大的深度为多少，None表示不限制
min_samples_split=2, -> 剪枝参数，表示在构建决策树的时候，如果当前数据集中的样本数目小于该值，那么对于当前数据集不进行划分。
min_samples_leaf=1, -> 剪枝参数，指定在叶子节点中只要要求具有多少样本，可选值类型：int和float，int表示数量，float表示占比
max_features=None, -> 在寻找最优划分的时候，从多少个特征属性中找最优划分，默认为None，表示在所有特征属性中选择最优划分
random_state=None, -> 随机数种子
max_leaf_nodes=None, -> 剪枝参数：最优允许的叶子节点数目，默认为None，表示不限制
"""
algo = DecisionTreeClassifier(criterion='gini', max_depth=2)

# 7. 模型的训练
algo.fit(x_train, y_train)

# 8. 模型效果评估
train_predict = algo.predict(x_train)
test_predict = algo.predict(x_test)
print("测试集上的效果(准确率):{}".format(algo.score(x_test, y_test)))
print("训练集上的效果(准确率):{}".format(algo.score(x_train, y_train)))
print("测试集上的效果(分类评估报告):\n{}".format(classification_report(y_test, test_predict)))
print("训练集上的效果(分类评估报告):\n{}".format(classification_report(y_train, train_predict)))
print("各个特征的重要性权重系数（值越大，对应的特征属性就越重要）：{}".format(algo.feature_importances_))

# 9. 其它
print("返回的预测概率值:\n{}".format(algo.predict_proba(x_test)))

# 10. 可视化
# b. 可视化方式二：直接使用pydotpuls库将数据输出为pdf或者png格式
from sklearn import tree
import pydotplus

dot_data = tree.export_graphviz(decision_tree=algo, out_file=None,
                                feature_names=['A', 'B', 'C', 'D'],
                                class_names=['0', '1', '2'],
                                filled=True, rounded=True,
                                special_characters=True
                                )

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('iris1.png')

# 测试一下
print("*" * 100)
x_test2 = x_test.iloc[:2, :]
print(x_test2)
# apply方法返回的是叶子节点下标
print(algo.apply(x_test2))
# predict返回对应叶子节点中多数类别的值
print(algo.predict(x_test2))
