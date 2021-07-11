#_*_ coding:utf-8_*_
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']    # 设置字体
mpl.rcParams['axes.unicode_minus'] = False      # 中文显示

x=np.arange(0.05,3,0.05)
# 常函数
y1=[5 for i in x]
plt.plot(x,y1,linewidth=2,label='常函数：y=5')
# 一次函数
y2=[2*i+1 for i in x]
plt.plot(x,y2,linewidth=2,label='一次函数：y=2x+1')
# 二次函数
y3=[1.5*i*i-3*i+1 for i in x]
plt.plot(x,y3,linewidth=2,label='二次函数：y=1.5$x^2$-3x+1')
# 幂函数
y4=[math.pow(i,2) for i in x]
plt.plot(x,y4,linewidth=2,label='幂函数：y=$x^2$')
# 指数函数
y5=[math.pow(2,i) for i in x]
plt.plot(x,y5,linewidth=2,label='指数函数：y=$2^x$')
# 对数函数
y6=[math.log(i,1.5) for i in x]
plt.plot(x,y6,linewidth=2,label='对数函数：$y=log1.5(x)$')

plt.legend(loc='best')
plt.grid(True)
plt.show()
