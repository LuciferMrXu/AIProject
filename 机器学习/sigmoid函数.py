#_*_ coding:utf-8_*_
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(h):
    return 1.0/(1.0+np.exp(-h))

# 定义x的范围，像素为0.1
h=np.arange(-10,10,0.1)
# sigmoid为上面定义的函数
s_h=sigmoid(h)
plt.plot(h,s_h)

# 在坐标轴上加一条竖直的线，0.0为竖直线在坐标轴上的位置
plt.axvline(0.0,color='k')
# 加水平间距通过坐标轴
plt.axhspan(0.0,1.0,facecolor='1.0',alpha=1.0,ls='dotted')
# 加水平线通过坐标轴
plt.axhline(y=0.5,ls='dotted',color='k')
# 加y轴刻度
plt.yticks([0.0,0.5,1.0])
# 加y轴范围
plt.ylim(-0.1,1.1)

plt.xlabel('h')
plt.ylabel('$S(h)$')
plt.show()

