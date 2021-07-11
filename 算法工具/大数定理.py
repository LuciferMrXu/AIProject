#_*_ coding:utf-8_*_
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random
# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 给定随机数的种子
random.seed(28)    # 使多次生成的随机数相同

def generate_random_int(n):
    '''产生n个1到9的随机数'''
    return [random.randint(1,9) for i in range(n)]

if __name__=='__main__':
    number = int(input('请输入抽取样本的次数：'))
    x=[i for i in range(number+1) if i != 0]
    # 产生number个[1,9]的随机数
    total_random_int=generate_random_int(number)
    # 求n个[1,9]的随机数的均值，n为自然数
    y=[np.mean(total_random_int[0:i+1]) for i in range(number)]

    # 绘图
    plt.plot(x, y, 'b-')
    plt.xlim(0, number)
    plt.grid(True)
    plt.show()