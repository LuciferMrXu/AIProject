# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/15
"""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


def predict(x, theta, intercept=0.0):
    """
    计算对应样本的预测值
    :param x:
    :param theta:
    :param intercept:
    :return:
    """
    result = 0.0
    # 1. x和theta相乘
    # np.dot(np.asarray(x).reshape(-1), np.asarray(theta).reshape((-1, 1)))
    n = len(x)
    for i in range(n):
        result += x[i] * theta[i]
    # 2. 加上截距项
    result += intercept
    # 3. 返回结果
    return result


def predict_X(X, theta, intercept=0.0):
    Y = []
    for x in X:
        Y.append(predict(x, theta, intercept))
    return Y


def fit(X, Y, alpha=0.0001, max_iter=None, tol=1e-5, fit_intercept=True):
    """
    进行模型训练，并将训练结果返回(返回模型参数)
    :param X:  输入的特征矩阵
    :param Y:  输入的目标矩阵
    :param alpha:  学习率
    :param max_iter:  最大迭代次数
    :param tol:  收敛条件
    :param fit_intercept: 是否训练截距项
    :return:
    """
    # 1. 对输入的数据进行整理
    X = np.asarray(X)
    Y = np.asarray(Y).reshape(-1)
    if max_iter is None:
        max_iter = 2000
    max_iter = max_iter if max_iter > 0 else 2000

    # 2. 获取样本数目、维度特征属性数目
    m, n = np.shape(X)
    if m != np.shape(Y)[0]:
        raise Exception("样本数据格式不统一!!!")

    # 3. 定义相关的模型参数以及变量
    theta = np.zeros(shape=[n])
    intercept = 0.0
    # 定义一个上一次的损失函数值
    pred_j = 1 << 64

    # 4. 开始模型迭代
    for k in range(max_iter):
        # a. 将更新的样本顺序打乱
        random_index = np.random.permutation(m)
        # b. 遍历所有样本，使用每个样本更新模型参数
        for index in random_index:
            # a. 计算当前样本的实际值和预测值之间的差值
            y_true = Y[index]
            y_predict = predict(X[index], theta, intercept)
            diff = y_true - y_predict

            # b. 基于计算出来的差值更新theta值
            for j in range(n):
                # i. 计算梯度值
                gd = diff * X[index][j]
                # ii. 基于计算的梯度值，更新theta值
                theta[j] = theta[j] + alpha * gd

            # c. 基于计算出来的差值更新截距项
            if fit_intercept:
                # i. 计算所有样本的累计梯度值
                gd = diff
                # ii. 更新截距项
                intercept = intercept + alpha * gd

        # d. 基于更新好的参数，重新计算一下损失函数的值
        sum_j = 0.0
        for i in range(m):
            y_true = Y[i]
            y_predict = predict(X[i], theta, intercept)
            sum_j += math.pow(y_true - y_predict, 2)
        sum_j /= m

        # e. 根据当前的损失函数值和上一个迭代的损失函数值之间的差值决定是否跳出训练
        if np.abs(pred_j - sum_j) < tol:
            break
        pred_j = sum_j

    # 5. 返回结果
    print("迭代{}次后，损失函数值为:{}".format(k, pred_j))
    return theta, intercept, pred_j


if __name__ == '__main__':
    # 1. 构建模拟数据
    np.random.seed(28)
    N = 10
    x = np.linspace(start=0, stop=6, num=N) + np.random.randn(N)
    y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.randn(N)
    x.shape = -1, 1
    y.shape = -1

    # 2. 使用算法模型构建
    # a. 使用LinearRegression
    algo1 = LinearRegression()
    algo1.fit(x, y)
    s1 = r2_score(y, algo1.predict(x))
    print("LinearRegression算法的模型参数:{}".format(algo1.coef_))
    print("LinearRegression算法的截距项:{}".format(algo1.intercept_))
    print("LinearRegression算法的评估指标:{}".format(s1))
    # b. 使用SGDRegressor
    algo2 = SGDRegressor(n_iter=1000)
    algo2.fit(x, y)
    s2 = r2_score(y, algo2.predict(x))
    print("SGDRegressor算法的模型参数:{}".format(algo2.coef_))
    print("SGDRegressor算法的截距项:{}".format(algo2.intercept_))
    print("SGDRegressor算法的评估指标:{}".format(s2))
    # c. 使用自己实现的算法进行模型训练
    theta, intercept, pred_j = fit(x, y, alpha=0.001, max_iter=2000, tol=1e-16)
    s3 = r2_score(y, predict_X(x, theta, intercept))
    print("自定义的线性回归算法的模型参数:{}".format(theta))
    print("自定义的线性回归算法的截距项:{}".format(intercept))
    print("自定义的线性回归算法的评估指标:{}".format(s3))

    # 构建画图用的模拟数据
    x_test = np.linspace(x.min(), x.max(), num=100)
    x_test.shape = -1, 1
    y_predict1 = algo1.predict(x_test)
    y_predict2 = algo2.predict(x_test)
    y_predict3 = predict_X(x_test, theta, intercept)

    # 可视化看一下
    plt.plot(x, y, 'ro', ms=10)
    plt.plot(x_test, y_predict1, color='#3568AA', lw=5, label=u'最小二乘的线性回归模型:%.3f' % s1)
    plt.plot(x_test, y_predict2, color='#b624db', lw=3, label=u'随机梯度下降的线性回归模型:%.3f' % s2)
    plt.plot(x_test, y_predict3, color='#FF0000', lw=2, label=u'自定义的线性回归模型:%.3f' % s3)
    plt.legend(loc='upper left')
    plt.suptitle(u'回归算法模型比较')
    plt.show()
