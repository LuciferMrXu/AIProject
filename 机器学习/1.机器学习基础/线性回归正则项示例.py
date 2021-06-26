#_*_ coding:utf-8_*_
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

# 三维原始图像
def f2(x, y, z):
    return (0.6 * x + 0.8 * y - z) ** 2


# 构建数据
X1 = np.arange(-4, 4.5, 0.2)
X2 = np.arange(-4, 4.5, 0.2)
X1, X2 = np.meshgrid(X1, X2)
Y = np.array(list(map(lambda t: f2(t[0], t[1],(0.75*t[0]+1.65*t[1]+np.random.randn())/10), zip(X1.flatten(), X2.flatten()))))
# 正则项/惩罚项为 zz=0.75*t[0]+1.65*t[1]+np.random.randn()
Y.shape = X1.shape

# 画图
fig = plt.figure(facecolor='w')
ax = Axes3D(fig)
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap=plt.cm.jet)


plt.show()
