# -- encoding:utf-8 --
"""
Create by ibf on 2018/6/30
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.metrics import f1_score, recall_score

# 1. 加载数据
# file_path = 'C:/workspace/python/sklearn07/datas/iris.data'
# file_path = 'file:///C:/workspace/python/sklearn07/datas/iris.data'
file_path = './datas/iris.data'
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'cla']
df = pd.read_csv(file_path, header=None, names=names)


# print(df.head())
# print(df.info())


# 2. 提取数据
def parse_record(record):
    result = []
    r = zip(names, record)
    for name, value in r:
        if name == 'cla':
            if value == 'Iris-setosa':
                result.append(1)
            elif value == 'Iris-versicolor':
                result.append(2)
            else:
                result.append(3)
        else:
            result.append(value)
    return result


datas = df.apply(lambda r: pd.Series(parse_record(r), index=names), axis=1)
# print(datas.head(10))
# print(datas.cla.value_counts())
x = datas[names[0:-1]]
y = datas[names[-1]]

# 3. 数据的分割
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9,test_size=0.1, random_state=28)

# 4. KNN算法模型的构建
# n_neighbors: 邻居数，默认为5; 一般需要调整的参数
# weights: 做预测的时候采用何种预测方式，是等权重还是不同权重
# algorithm：模型训练过程中的采用方式
# leaf_size: 构建Tree的过程中，停止构建的条件，最多的叶子数目
# p&metric: 样本相似度度量方式，默认为欧几里得距离
knn = KNeighborsClassifier()


# 构建网格参数选择的交叉验证对象
# estimator: 给定对哪个模型算法对象进行参数调优
# param_grid: 给定estimator对应的模型可选的参数列表有哪些，是一个dict字典数据类型
# param_grid中的key是estimator算法对应的参数,是字符串形式；value是该可参数可选的值列表集合
param_grid = {
    'n_neighbors': [2, 5, 10],
    'weights': ['uniform', 'distance'],
    'algorithm': ['kd_tree', 'brute','ball_tree'],
    'leaf_size': [5, 10, 20]
}
algo = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)

# 5. 算法模型的训练
algo.fit(x_train, y_train)

# 获取实际最优模型
print("最优模型:{}".format(algo.best_estimator_))
print("最优模型对应的参数:{}".format(algo.best_params_))

# 6. 模型效果评估
# 看方法的注释的方法：按ctrl键，同时鼠标在方法上左键点击
print("训练集上的准确率:{}".format(algo.best_estimator_.score(x_train, y_train)))
print("测试集上的准确率:{}".format(algo.best_estimator_.score(x_test, y_test)))
print("训练集上的F1值:{}".format(f1_score(y_train, algo.predict(x_train), average='macro')))
print("测试集上的F1值:{}".format(f1_score(y_test, algo.predict(x_test), average='macro')))
print("训练集上的召回率:{}".format(recall_score(y_train, algo.predict(x_train), average='macro')))
print("测试集上的召回率:{}".format(recall_score(y_test, algo.predict(x_test), average='macro')))

# 7. 模型保存（保存最优模型）
joblib.dump(algo.best_estimator_, 'G:/MongoDB/model/KNN_iris.m')
