#_*_ coding:utf-8_*_
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import numpy as np
# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

'''
随机的抛六面的骰子，计算三次的点数的和, 三次点数的和其实就是一个事件A
==> 事件A的发生属于什么分布？
==> A=x1+x2+x3，其中x1，x2，x3分别三次的抛骰子的点数
根据中心极限定理，由于x1，x2，x3属于独立同分布的，所以说最终的事件A属于高斯分布
'''

def generate_random_int():
    '''随机产生一个1到6的整数，表示是一个六面骰子的结果'''
    return random.randint(1,6)

def generate_sum(n):
    '''计算返回n次抛六面骰子和的结果'''
    return np.sum([generate_random_int() for i in range(n)])

if __name__=='__main__':
    a = int(input('请输入进行多少轮实验：'))
    b = int(input('请输入每一轮抛几次骰子：'))
    # 进行A事件多少次
    number1=a
    # 表示每次A事件抛几次骰子
    number2=b

    # 进行number1次事件A的操作，每次事件A都进行number2次抛骰子
    keys=[generate_sum(number2) for i in range(number1)]

    # 统计每个sum数字出现的次数
    result={}
    for key in keys:
        count=1
        if key in result:
            count+=result[key]
        result[key]=count

    # 获取x和y
    x=sorted(np.unique(list(result.keys())))   # 对result去重
    y=[]
    for key in x:
        # 将出现的次数进行百分比计算
        y.append(result[key]/number1)

    # 绘图
    plt.plot(x,y,'b-')
    plt.xlim(x[0]-1,x[-1]+1)
    plt.grid(True)
    plt.show()