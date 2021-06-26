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
path = '../datas/iris.data'
names = ['A', 'B', 'C', 'D', 'cla']
df = pd.read_csv(path, header=None, names=names)


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
                result.append(0)
        else:
            result.append(value)
    return result


df = df.apply(lambda row: pd.Series(parse_record(row), index=names), axis=1)
df.cla = df.cla.astype(np.int32)
df.info()
# print(df.cla.value_counts())
flag = False
# df = df[df.cla != 3]
# print(df.cla.value_counts())

# # 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
X = df[names[0:1]]
Y = df[names[-1]]

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
algo = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# 7. 模型的训练
algo.fit(x_train, y_train)

# 8. 模型效果评估
train_predict = algo.predict(x_train)
test_predict = algo.predict(x_test)
print("测试集上的效果(准确率):{}".format(algo.score(x_test, y_test)))
print("训练集上的效果(准确率):{}".format(algo.score(x_train, y_train)))

# 9. 调整阈值
print("参数值theta:\n{}".format(algo.coef_))
print("参数值截距项:\n{}".format(algo.intercept_))
print(test_predict)
print("decision_function API返回值:\n{}".format(algo.decision_function(x_test)))
print("softmax/sigmoid函数返回的概率值:\n{}".format(algo.predict_proba(x_test)))
print("softmax/sigmoid函数返回的经过对数转换之后的概率值:\n{}".format(algo.predict_log_proba(x_test)))

# 画图看一下
x_min = np.min(X).astype(np.float32) - 0.5
x_max = np.max(X).astype(np.float32) + 0.5
# TODO: 自己考虑如果是多分类线条是什么样子
if flag:
    theta1 = algo.coef_[0][0]
    theta0 = algo.intercept_[0]

    plt.plot(x_test, y_test, 'ro', markersize=6, zorder=10, label=u'真实值')
    plt.plot(x_test, test_predict, 'go', markersize=10, label=u'预测值')
    # plt.plot(x_test, theta1 * x_test + theta0, 'go', markersize=10, label=u'预测值')
    plt.plot([x_min, x_max], [theta1 * x_min + theta0, theta1 * x_max + theta0], 'r-', linewidth=3, label=u'第一条直线')
    plt.plot([x_min, x_max], [0, 0], 'g--')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
else:
    theta11, theta21, theta31 = algo.coef_[:, 0]
    theta10, theta20, theta30 = algo.intercept_

    plt.plot(x_test, y_test, 'ro', markersize=6, zorder=10, label=u'真实值')
    plt.plot(x_test, test_predict, 'go', markersize=10, label=u'预测值')
    # plt.plot(x_test, theta1 * x_test + theta0, 'go', markersize=10, label=u'预测值')
    plt.plot([x_min, x_max], [theta11 * x_min + theta10, theta11 * x_max + theta10], 'r-', linewidth=3, label=u'第一条直线')
    plt.plot([x_min, x_max], [theta21 * x_min + theta20, theta21 * x_max + theta20], 'b-', linewidth=3, label=u'第二条直线')
    plt.plot([x_min, x_max], [theta31 * x_min + theta30, theta31 * x_max + theta30], 'g-', linewidth=3, label=u'第三条直线')
    plt.plot([x_min, x_max], [0, 0], 'g--')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
