#_*_ coding:utf-8_*_
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def validate(X, Y):
    """
    校验X和Y的格式是否正常，如果不正常，返回False；否则返回True
    :param X:
    :param Y:
    :return:
    """
    m1, n1 = np.shape(X)
    m2, n2 = np.shape(Y)
    if m1 != m2:
        return False
    else:
        if n2 != 1:
            return False
        else:
            return True


def predict(x, theta, intercept=0.0):
    """
    计算预测值
    :param x:
    :param theta:
    :param intercept:
    :return:
    """
    result = 0.0
    # 1. x和theta相乘
    n = len(x)
    for i in range(n):
        result += x[i] * theta[i]
    # 2. 加上截距项
    result += intercept
    # 返回结果
    return result


def predict_X(X, theta, intercept=0.0):
    Y = []
    for x in X:
        Y.append(predict(x, theta, intercept))
    return Y


def fit(X, Y, alpha=0.01, max_iter=-1, tol=1e-4, fit_intercept=True):
    """
    进行模型训练，方法返回模型训练得到的θ值以及截距项
    :param X: 输入的特征矩阵x，格式为二维数组形式，m*n, m表示样本数目，n表示每个样本的特征维度数目
    :param Y: 输入的目标矩阵y，格式为二维数组形式，m*1，m表示样本数目，1表示每个样本有一个待预测的特征
    :param alpha: 梯度下降中的步长或者学习率
    :param max_iter: 梯度下降迭代中最大迭代次数(数据从头到尾处理一遍，认为是一个迭代)
    :param tol:  收敛变换最小值
    :param fit_intercept:  是否训练模型的截距项
    :return:
    """
    # 1. 校验一下X和Y的格式是否正常
    assert validate(X, Y)

    X = np.array(X)
    Y = np.array(Y)
    # 2. 开始获取相关参数
    # 获取样本数和维度数目
    m, n = np.shape(X)
    # 定义theta参数
    theta = np.zeros(shape=[n])
    # 定义截距项
    intercept = 0
    # 获取最大允许迭代次数
    max_iter = 200 if max_iter <= 0 else max_iter
    # 构建一个误差保存的对象
    diff = np.zeros(shape=[m])
    # 定义一个之前状态的损失函数值
    pre_sum_j = 1 << 32

    # 3. 开始迭代，获取模型参数
    for i in range(max_iter):
        """
        作业只实现BGD的算法
        """
        # a. 在当前的theta取值的情况下，计算各个样本的预测值和实际值之间的差值(误差error)
        for k in range(m):
            y_true = Y[k][0]
            y_predict = predict(X[k], theta, intercept)
            diff[k] = y_true - y_predict

        # b. 对每个θ遍历求解
        for j in range(n):
            # 求解梯度值
            gd = 0
            for k in range(m):
                gd += diff[k] * X[k][j]
            # 进行参数模型更新
            theta[j] += alpha * gd

        # c. 对截距项求解
        if fit_intercept:
            # 求解梯度值
            # gd = 0
            # for k in range(m):
            #     gd += diff[k] * 1
            gd = np.sum(diff)
            # 进行参数模型更新
            intercept += alpha * gd

        # d. 计算在当前模型参数的情况下的误差值
        sum_j = 0.0
        for k in range(m):
            y_true = Y[k][0]
            y_predict = predict(X[k], theta, intercept)
            current_j = y_true - y_predict
            sum_j += math.pow(current_j, 2)
        sum_j /= m
        # 保存之前的最小损失函数值
        tmp_pre_sum_j = pre_sum_j
        pre_sum_j = sum_j

        # 当两次的误差损失函数的值变化小于tol的时候，直接结束模型训练
        if np.abs(tmp_pre_sum_j - sum_j) < tol:
            break

    # 函数返回值
    print("迭代{}次后，损失函数值为:{}".format(i, pre_sum_j))
    return theta, intercept, pre_sum_j


def score_X_Y(X, Y, theta, intercept=0.0):
    """
    计算R^2
    :param X:
    :param Y:
    :param theta:
    :param intercept:
    :return:
    """
    # 1. 获取预测值
    y_predict = predict_X(X, theta, intercept)
    # 2. 获取R^2
    r2 = score(Y, y_predict)
    # 3. 结果返回
    return r2


def score(y_true, y_predict):
    # 1. 计算rss和tss
    average_y_true = np.average(y_true)
    m = len(y_true)
    rss = 0.0
    tss = 0.0
    for k in range(m):
        rss += math.pow(y_true[k] - y_predict[k], 2)
        tss += math.pow(y_true[k] - average_y_true, 2)
    # 2. 计算R2
    r2 = 1.0 - 1.0 * rss / tss
    # 3. 返回结果
    return r2


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 构建模拟数据
np.random.seed(0)
N = 10
x = np.linspace(start=0, stop=6, num=N) + np.random.randn(N)
y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.randn(N)
x.shape = -1, 1
y.shape = -1, 1
# y = y.reshape(-1, 1)
# y = np.reshape(y, (-1, 1))
print(x.shape)
print(y.shape)

# 2. 算法构建
# a. 使用sklearn中的算法
model = LinearRegression()
model.fit(x, y)
# s1 = model.score(x, y)
# s1 = score_X_Y(x, y, model.coef_, model.intercept_)
s1 = score(y, model.predict(x))
print("模块自带线性回归默认实现模型=====================")
print("模型自带score API评估值为:{}".format(s1))
print("参数列表为:{}".format(model.coef_))
print("截距项为:{}".format(model.intercept_))

# b. 使用自定义的梯度下降线性回归算法进行模型构建
theta, intercept, sum_j = fit(x, y)
s2 = score_X_Y(x, y, theta, intercept)
print("自定义的线性回归模型===========================")
print("自定义模型自带score评估值为:{}".format(s2))
print("参数为：{}".format(theta))
print("截距项为:{}".format(intercept))

# 构建画图用的模拟数据
x_hat = np.linspace(x.min(), x.max(), num=100)
x_hat.shape = -1, 1
y_hat = model.predict(x_hat)
y_hat2 = predict_X(x_hat, theta, intercept)

# 画图看一下
plt.plot(x, y, 'ro', ms=10)
plt.plot(x_hat, y_hat, color='#b624db', lw=2, label=u'sklearn线性模型，$R^2$:%.3f' % s1)
plt.plot(x_hat, y_hat2, color='#0049b6', lw=2, label=u'自定义线性模型，$R^2$:%.3f' % s2)
plt.legend(loc='upper left')
plt.xlabel('X')
plt.ylabel('Y')
plt.suptitle(u'sklearn模型和自定义模型的效果比较', fontsize=20)
plt.grid(True)
plt.show()
