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


if __name__ == '__main__':
    x = np.arange(0, 1.0, step=0.05)
    y = 3.0 + 1.7 * x + 0.1 * np.random.normal(0.0, 1.0, size=len(x))
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    ones = np.ones_like(x)
    x = np.concatenate([ones, x], 1)
    print(x.shape)
    print(y.shape)

    theta = regres(x, y)

    fit = plt.figure()
    ax = fit.add_subplot(111)
    ax.scatter(x[:, 1], y[:, 0], s=20, c='r')
    x_mat = np.mat(x)
    x_mat.sort(0)
    y_predict = x_mat * theta
    ax.plot(x_mat[:,1], y_predict, linewidth=3)
    plt.show()

