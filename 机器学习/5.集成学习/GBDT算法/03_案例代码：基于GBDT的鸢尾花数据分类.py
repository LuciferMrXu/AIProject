# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/11
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
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
n_estimators=10, 最终训练的子模型的数量
criterion="gini", 随机森林底层的决策树使用什么指标衡量数据的纯度，默认gini系数，可选gini和entropy
max_depth=None, 底层决策树允许的最大深度，默认不限制
min_samples_split=2, 底层决策树分裂的时候，要求数据集中至少的样本数
min_samples_leaf=1, 底层决策树分类的时候，每个叶子节点中样本数至少要求的样本数
max_features="auto", 底层决策树构建过程中，选择特征属性的时候，选择最优特征是从多少个原始特征中选择的，默认为auto
max_leaf_nodes=None, 底层决策树构建后，最多允许的叶子节点数目
bootstrap=True, 在对于每个子模型训练的时候，是否使用有放回的重采样产生数据
oob_score=False, 是否计算袋外准确率(没有参与模型训练的数据称为oob数据<袋外数据>，其实就是当模型构建好之后，将oob数据输入到模型看一下效果) -> 只有当进行有放回的重采样的时候才有这个值，只有当bootstrap为True时，该值才可以设置为True
"""
algo = GradientBoostingClassifier(n_estimators=10, max_depth=3)

# 7. 模型的训练
algo.fit(x_train, y_train)

# 8. 模型效果评估
train_predict = algo.predict(x_train)
test_predict = algo.predict(x_test)
print("测试集上的效果(准确率):{}".format(algo.score(x_test, y_test)))
print("训练集上的效果(准确率):{}".format(algo.score(x_train, y_train)))
print("测试集上的效果(分类评估报告):\n{}".format(classification_report(y_test, test_predict)))
print("训练集上的效果(分类评估报告):\n{}".format(classification_report(y_train, train_predict)))

# 9. 其它
print("返回的预测概率值:\n{}".format(algo.predict_proba(x_test)))

# 10. 其他特殊的API
print("子模型列表:\n{}".format(algo.estimators_))
print("各个特征属性的重要性权重:\n{}".format(algo.feature_importances_))

from sklearn import tree
import pydotplus

k = 0
for algo1 in algo.estimators_:
    kk = 0
    for algo2 in algo1:
        dot_data = tree.export_graphviz(decision_tree=algo2, out_file=None,
                                        feature_names=['A', 'B', 'C', 'D'],
                                        class_names=['1', '2', '3'],
                                        filled=True, rounded=True,
                                        special_characters=True
                                        )

        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png('rgdt_{}_{}.png'.format(k, kk))
        kk += 1
    k += 1
    if k > 3:
        break

# 返回叶子节点下标
print("*" * 100)
x_test2 = x_test.iloc[:2, :]
print(x_test2)
# apply方法返回的是叶子节点下标
print(algo.apply(x_test2))
