# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/17
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 设置一下，防止乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(28)

# 产生模拟数据
x = 0.3 * np.random.rand(100, 2)
x_train = np.vstack([x + 2, x - 2])
x = 0.3 * np.random.rand(20, 2)
x_test = np.vstack([x + 2, x - 2])
x_outliers = np.random.uniform(low=-2.5, high=2.5, size=(20, 2))

# 模型构建
algo = IsolationForest(n_estimators=100,contamination="auto",behaviour='new')
algo.fit(x_train)

# 模型预测(1表示正常样本，-1表示异常样本)
y_pred_train = algo.predict(x_train)
print(y_pred_train)


y_pred_test = algo.predict(x_test)
print(y_pred_test)
y_pred_outliers = algo.predict(x_outliers)
print(y_pred_outliers)

# 看一下decision_function
'''
    decision_function是决策值:θ^T*x
'''
print(algo.decision_function(x_test))
print(algo.decision_function(x_outliers))



# 画图可视化
x1_min = -3
x1_max = 3
x2_min = -3
x2_max = 3

# 等距离的从最小值到最大值之间产生50点
t1 = np.linspace(x1_min, x1_max, 50)
t2 = np.linspace(x2_min, x2_max, 50)
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
x_show = np.dstack((x1.flat, x2.flat))[0]  # 测试点
z = algo.decision_function(x_show)
z = z.reshape(x1.shape)


# 画一个等高线区域图
plt.contourf(x1, x2, z, cmap=plt.cm.Blues_r)
plt.scatter(x_train[:, 0], x_train[:, 1], c='b')
plt.scatter(x_test[:, 0], x_test[:, 1], c='g')
plt.scatter(x_outliers[:, 0], x_outliers[:, 1], c='r')
plt.show()


from sklearn import tree
import pydotplus

k = 0
for algo1 in algo.estimators_:
    dot_data = tree.export_graphviz(decision_tree=algo1, out_file=None,
                                    feature_names=['A', 'B'],
                                    class_names=['1', '2', '3'],
                                    filled=True, rounded=True,
                                    special_characters=True
                                    )

    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('isolation_{}.png'.format(k))
    k += 1
    if k > 3:
        break
