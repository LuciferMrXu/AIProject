#_*_ coding:utf-8_*_
import numpy as np
from wzx.hmm import viterbi
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics.pairwise import pairwise_distances_argmin
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def expand(a, b):
    d = (b - a) * 0.05
    return a-d, b+d

def load(path):
    x = np.loadtxt(path, delimiter='\t', skiprows=2, usecols=(4, 5, 6, 2, 3))
    x = x[:-1, :]  # 最后一天的数据不考虑
    close_price = x[:, 0]  # 收盘价
    volumn = x[:, 1]  # 成交量
    amount = x[:, 2]  # 成交额
    amplitude_price = x[:, 3] - x[:, 4]  # 每天的最高价与最低价的差
    diff_price = np.diff(close_price)  # 涨跌值(每天相对于昨天的涨跌幅)
    volumn = volumn[1:]  # 成交量(今天的成交量)
    amount = amount[1:]  # 成交额(今天的成交额度)
    amplitude_price = amplitude_price[1:]  # 每日振幅(今天的振幅)

    # 相当于整个数据相当于一个序列，序列中的每个样本具有四个特征
    sample = np.column_stack((volumn, amount, amplitude_price, diff_price))  # 观测值
    print("样本数目:%d, 每个样本的特征数目:%d" % sample.shape)
    print(sample)
    return sample


if __name__=='__main__':
    path='./datas/SH600000.txt'
    load(path)