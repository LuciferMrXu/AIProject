# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/10
"""

import numpy as np
import matplotlib.pyplot as plt


def regres(x_arr, y_arr):
    xmat = np.mat(x_arr)
    ymat = np.mat(y_arr)

    xTx = xmat.T * xmat
    if np.linalg.det(xTx) == 0.0:
        print("ERROR")
        return
    xTx = xTx.I

    theta = xTx * (xmat.T * ymat)
    return theta


def lwlr(test_point, x_arr, y_arr, k=1.0):
    xmat = np.mat(x_arr)
    ymat = np.mat(y_arr)
    weights = np.mat(np.eye(np.shape(xmat)[0]))

    # 更新weights值
    for j in range(np.shape(xmat)[0]):
        diffmat = test_point - xmat[j, :]
        weights[j, j] = np.exp(diffmat * diffmat.T / (-2 * k ** 2))

    xTx = xmat.T * (weights * xmat)
    if np.linalg.det(xTx) == 0.0:
        print("ERROR")
        return
    xTx = xTx.I

    theta = xTx * (xmat.T * (weights * ymat))
    return theta, test_point * theta


def lwlr_predict(test_x_arr, x_arr, y_arr, k=1.0):
    test_x_arr = np.mat(test_x_arr)
    m = np.shape(test_x_arr)[0]
    y_predict = np.zeros(m)
    for i in range(m):
        _, y_predict[i] = lwlr(test_x_arr[i, :], x_arr, y_arr, k)
    return y_predict


if __name__ == '__main__':
    x = np.arange(0, 1.0, step=0.005)
    y = 3.0 + 1.7 * x + 0.1 * np.sin(60 * x) + 0.02 * np.random.normal(0.0, 1.0, size=len(x))
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    ones = np.ones_like(x)
    x = np.concatenate([ones, x], 1)
    print(x.shape)
    print(y.shape)
    flag = False

    x_mat = np.mat(x)
    x_mat.sort(0)
    if flag:
        print("使用基础的线性回归")
        theta = regres(x, y)
        y_predict = x_mat * theta
    else:
        print("使用局部加权线性回归")
        y_predict = lwlr_predict(x_mat, x, y, 0.05)

    fit = plt.figure()
    ax = fit.add_subplot(111)
    ax.scatter(x[:, 1], y[:, 0], s=20, c='r')

    ax.plot(x_mat[:, 1], y_predict, linewidth=3)
    plt.show()
