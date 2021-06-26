# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/18
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score

np.random.seed(28)
flag = 1
if flag == 1:
    # 回归的实现
    x = np.random.randn(10, 2) * 5
    y = np.random.randn(10) * 3
    y_true = y

    # 使用简单的决策树模型看一下效果
    algo = DecisionTreeRegressor(max_depth=1)
    algo.fit(x, y)
    print("单模型训练数据集上效果:{}".format(r2_score(y_true, algo.predict(x))))
    print("实际y值:{}".format(y_true))
    print("预测y值:{}".format(algo.predict(x)))

    # GBDT回归算法的构建过程
    models = []
    # 构建第一个模型
    m1 = np.mean(y)
    models.append(m1)
    # 构建后面的模型
    learn_rate = 0.1
    pred_m = m1
    n = 1000
    for i in range(n):
        # 计算负梯度值，也就是更新y值
        if i == 0:
            y = y - learn_rate * pred_m
        else:
            y = y - learn_rate * pred_m.predict(x).reshape(y.shape)
        # 基于新的y值训练回归决策树
        model = DecisionTreeRegressor(max_depth=1)
        model.fit(x, y)
        models.append(model)
        pred_m = model

    print("模型构建完成，总模型数目:{}".format(len(models)))
    print("开始预测")
    y_pred = np.zeros_like(y)
    for i in range(n + 1):
        model = models[i]
        if i == 0:
            y_pred = y_pred + learn_rate * model
        else:
            y_pred = y_pred + learn_rate * model.predict(x).reshape(y.shape)
    print("GBDT效果:{}".format(r2_score(y_true, y_pred)))
    print("实际值:{}".format(y_true))
    print("预测值:{}".format(y_pred))
elif flag == 2:
    # 做一个二分类的
    x = np.random.randn(10, 2) * 5
    y = np.array([1] * 6 + [0] * 4).astype(np.int)
    y_true = y

    # 使用简单的决策树模型看一下效果
    algo = DecisionTreeClassifier(max_depth=1)
    algo.fit(x, y)
    print("单模型训练数据集上效果:{}".format(accuracy_score(y_true, algo.predict(x))))
    print("实际y值:{}".format(y_true))
    print("预测y值:{}".format(algo.predict(x)))

    # GBDT分类算法的构建过程
    models = []
    # 构建第一个模型
    m1 = np.log(6.0 / 4)
    models.append(m1)
    # 构建后面的模型
    learn_rate = 0.1
    pred_m = m1
    n = 1000
    for i in range(n):
        # 计算负梯度值，也就是更新y值
        if i == 0:
            y = y - learn_rate * pred_m
        else:
            y = y - learn_rate * pred_m.predict(x).reshape(y.shape)
        # 基于新的y值训练回归决策树
        model = DecisionTreeRegressor(max_depth=1)
        model.fit(x, y)
        models.append(model)
        pred_m = model

    print("模型构建完成，总模型数目:{}".format(len(models)))
    print("开始预测")
    y_pred = np.zeros_like(y)
    for i in range(n + 1):
        model = models[i]
        if i == 0:
            y_pred = y_pred + learn_rate * model
        else:
            y_pred = y_pred + learn_rate * model.predict(x).reshape(y.shape)
    y_hat = np.zeros_like(y_pred, np.int)
    y_hat[y_pred >= 0.5] = 1
    y_hat[y_pred < 0.5] = 0
    print("GBDT效果:{}".format(accuracy_score(y_true, y_hat)))
    print("实际值:{}".format(y_true))
    print("预测值:{}".format(y_hat))
    print("决策树函数值:{}".format(y_pred))
elif flag == 3:
    # 多分类
    x = np.random.randn(10, 2) * 5
    y = np.array([0] * 2 + [1] * 4 + [2] * 4).astype(np.int)
    y_true = y

    # 针对于每个类别构建一个y，属于当前类别设置为1，不属于设置为0
    y1 = np.array([1] * 2 + [0] * 8).astype(np.int)
    y2 = np.array([0] * 2 + [1] * 4 + [0] * 4).astype(np.int)
    y3 = np.array([0] * 6 + [1] * 4).astype(np.int)
    ys = [y1, y2, y3]

    # 使用简单的决策树模型看一下效果
    algo = DecisionTreeClassifier(max_depth=1)
    algo.fit(x, y)
    print("单模型训练数据集上效果:{}".format(accuracy_score(y_true, algo.predict(x))))
    print("实际y值:{}".format(y_true))
    print("预测y值:{}".format(algo.predict(x)))

    # GBDT分类算法的构建过程
    models = []
    # 构建第一个模型
    m1 = np.asarray([0, 0, 0])
    models.append(m1)
    # 构建后面的模型
    learn_rate = 0.1
    pred_m = m1
    n = 1000
    for i in range(n):
        # 构建一个保存第i次迭代时候产生的所有子模型
        tmp_algos = []
        # 计算负梯度值，也就是更新y值
        if i == 0:
            pred_y = pred_m
        else:
            pred_y = np.asarray(list(map(lambda algo: algo.predict(x), pred_m)))
        for k in range(len(ys)):
            p = np.exp(pred_y[k]) / np.sum(np.exp(pred_y))
            ys[k] = ys[k] - p

            # 基于新的y值训练回归决策树
            model = DecisionTreeRegressor(max_depth=1)
            model.fit(x, ys[k])
            tmp_algos.append(model)

        models.append(tmp_algos)
        pred_m = tmp_algos

    print("模型构建完成，总模型数目:{}".format(len(models) * len(ys)))
    print("开始预测")
    y_pred = np.zeros((len(ys), y.shape[0]))
    for i in range(n + 1):
        model = models[i]
        for k in range(len(ys)):
            if i == 0:
                y_pred[k] = model[k]
            else:
                y_pred[k] += model[k].predict(x).reshape(y.shape)

    y_hat = np.argmax(y_pred, axis=0).astype(np.int)
    print("GBDT效果:{}".format(accuracy_score(y_true, y_hat)))
    print("实际值:{}".format(y_true))
    print("预测值:{}".format(y_hat))
    print("决策树函数值:{}".format(y_pred))
