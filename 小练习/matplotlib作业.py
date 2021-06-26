#_*_ coding:utf-8_*_
import matplotlib.pyplot as plt
import numpy as np

def fun1(x):
    return np.log(x)
def fun2(x):
    return np.log10(x)
def fun3(x):
    return np.log(x)/np.log(5)
def fun4(x):
    return np.log(x)/np.log(0.5)

x=np.arange(0.1,3.1,0.001)

y1=[fun1(i) for i in x]
y2=[fun2(i) for i in x]
y3=[fun3(i) for i in x]
y4=[fun4(i) for i in x]

plt.figure()

plt.plot(x,y1,linewidth=2,color='b',label='$loge(x)$')
plt.plot(x,y2,linewidth=2,color='r',label='$log10(x)$')
plt.plot(x,y3,linewidth=2,color='g',label='$log5(x)$')
plt.plot(x,y4,linewidth=2,color='y',label='$log0.5(x)$')
plt.plot([1,1],[-3,5],'--',color='#999999',linewidth=2)

plt.ylim(-3,5)
plt.legend(loc='best')
plt.grid(True)
plt.show()
