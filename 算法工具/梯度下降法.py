#_*_ coding:utf-8_*_
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False


'''
由于梯度下降法中负梯度方向作为变量的变化方向，所以有可能导致最终求解的值是局部最优解，所以在使用梯度下降的时候，一般需要进行一些调优策略：
    1、学习率的选择：学习率过大，表示每次迭代更新的时候变化比较大，有可能会跳过
    最优解；学习率过小，表示每次迭代更新的时候变化比较小，就会导致迭代速度过
    慢，很长时间都不能结束；
    2、算法初始参数值的选择：初始值不同，最终获得的最小值也有可能不同，因为梯度
    下降法求解的是局部最优解，所以一般情况下，选择多次不同初始值运行算法，并
    最终返回损失函数最小情况下的结果值；
    3、标准化：由于样本不同特征的取值范围不同，可能会导致在各个不同参数上迭代速
    度不同，为了减少特征取值的影响，可以将特征进行标准化操作。
'''

'''
1、批量梯度下降法(Batch Gradient Descent, BGD)：使用所有样本在当前点的梯度值来对变量参数进行更新操作。
2、随机梯度下降法(Stochastic Gradient Descent, SGD)：在更新变量参数的时候，选取一个样本的梯度值来更新参数。
3、小批量梯度下降法(Mini-batch Gradient Descent, MBGD)：结合BGD和SGD的特性，从原始数据中，每次选择n个样本来更新参数值，一般n选择10.

    当样本量为m的时候，每次迭代BGD算法中对于参数值更新一次，SGD算法中对于
    参数值更新m次，MBGD算法中对于参数值更新m/n次，相对来讲SGD算法的更新
    速度最快；

    SGD算法中对于每个样本都需要更新参数值，当样本值不太正常的时候，就有可能
    会导致本次的参数更新会产生相反的影响，也就是说SGD算法的结果并不是完全收
    敛的，而是在收敛结果处波动的；可以跳过局部最优解。

    SGD算法是每个样本都更新一次参数值，所以SGD算法特别适合样本数据量大的情
    况以及在线机器学习(Online ML)。
'''


'''
    梯度下降案例 z=f(x)=x^2
'''
# 原函数
def f(x):
    return x**2
# 导数
def h(t):
    return 2*t

X=[]
Y=[]

x=int(input('请输入x的值：'))
step=0.8
f_change=f(x)
f_current=f(x)

X.append(x)
Y.append(f_current)

while f_change>1e-10:
    x = x - step * h(x)
    tmp=f(x)
    f_change=np.abs(f_current-tmp)
    f_current=tmp
    X.append(x)
    Y.append(f_current)

print('最终结果为：',(x,f_current))


# 绘图
fig=plt.figure()
X2=np.arange(-2.1,2.15,0.05)
Y2=X2**2

plt.plot(X2,Y2,'-',color='#666666',linewidth=2)
plt.plot(X,Y,'b--o')
plt.title('$y=x^2$函数求解最小值，最终解为：x=%.2f，y=%.2f'%(x,f_current))
plt.show()


'''
    梯度下降案例 z=f(x,y)=x^2+y^2
'''
# 原函数
def f(x,y):
    return x**2 + y**2
# 偏导数
def h(t):
    return 2*t

X=[]
Y=[]
Z=[]

x=int(input('请输入x的值：'))
y=int(input('请输入y的值：'))
f_change=x**2 + y**2
f_current=f(x,y)
step=0.1
X.append(x)
Y.append(y)
Z.append(f_current)

while f_change>1e-10:
    x = x - step * h(x)
    y = y - step * h(y)
    f_change=f_current-f(x,y)
    f_current=f(x,y)
    X.append(x)
    Y.append(y)
    Z.append(f_current)

print('最终结果为：',(x,y))



# 绘图
fig=plt.figure()
ax=Axes3D(fig)
X2=np.arange(-2,2,0.2)
Y2=np.arange(-2,2,0.2)
X2,Y2=np.meshgrid(X2,Y2)
Z2=X2**2+Y2**2

ax.plot_surface(X2,Y2,Z2,rstride=1,cstride=1,cmap='rainbow')
ax.plot(X,Y,Z,'r--o')

ax.set_title('梯度下降解法，最终解为：x=%.2f，y=%.2f，z=%.2f'%(x,y,f_current))
plt.show()
