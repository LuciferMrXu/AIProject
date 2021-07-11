#_*_ coding:utf-8_*_
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False


# 对数函数
x1=np.arange(0.05,3,0.05)
y1=[math.log(i,math.e) for i in x1]
y2=[math.log(i,10) for i in x1]
y3=[math.log(i,5) for i in x1]
y4=[math.log(i,0.5) for i in x1]
plt.plot(x1,y1,linewidth=2,color='b',label='$loge(x)$')
plt.plot(x1,y2,linewidth=2,color='r',label='$log10(x)$')
plt.plot(x1,y3,linewidth=2,color='g',label='$log5(x)$')
plt.plot(x1,y4,linewidth=2,color='y',label='$log0.5(x)$')
plt.plot([1,1],[-3,5],'--',color='#999999',linewidth=2)

plt.xlim(0,3)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 指数函数
x2=np.arange(-2,2,0.05)
y5=[math.pow(0.5,i) for i in x2]
y6=[math.pow(math.e,i) for i in x2]
y7=[math.pow(5,i) for i in x2]

plt.plot(x2,y5,linewidth=2,color='r',label='$0.5^x$')
plt.plot(x2,y6,linewidth=2,color='g',label='$e^x$')
plt.plot(x2,y7,linewidth=2,color='b',label='$5^x$')
plt.plot([0],[1],'o',color='#999999',linewidth=2)
plt.legend(loc='upper left')
plt.xlim(-2,2)
plt.grid(True)
plt.show()