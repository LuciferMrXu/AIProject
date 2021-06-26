# -- encoding:utf-8 --
"""
Create by ibf on 2018/7/17
"""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

n = 10
x = np.random.normal(loc=0, scale=100, size=(n, 2))
y = np.random.normal(loc=0, scale=1, size=(n, 1))


def f1(theta1, theta2, x1, x2, y):
    return math.pow(x1, 2) * math.pow(theta1, 2) + math.pow(x2, 2) * math.pow(theta2,2) \
           + 2 * theta1 * theta2 * x1 * x2 + math.pow(y, 2) - 2 * y * theta1 * x1 - 2 * y * theta2 * x2


def f(theta1, theta2):
    result = 0
    for tx1, tx2, ty in np.hstack((x, y)):
        result += f1(theta1, theta2, tx1, tx2, ty)
    return result


t1 = np.linspace(-10, 10, 100)
t2 = np.linspace(-10, 10, 100)
t1, t2 = np.meshgrid(t1, t2)
t3 = np.array(list(map(lambda t: f(t[0], t[1]), zip(t1.flatten(), t2.flatten()))))
t3.shape = t1.shape

fig = plt.figure(facecolor='w')
ax = Axes3D(fig)
ax.plot_surface(t1, t2, t3, rstride=1, cstride=1, cmap=plt.cm.jet)
plt.show()
