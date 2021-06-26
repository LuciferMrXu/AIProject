#coding:utf-8
from __future__  import division
import pandas as pd
import numpy as np



def maIndictor(Price, length, name = 'nums'):
    """此函数可以计算移动平均线，编写者：Allen
            指标简介：移动平均线是由著名的美国投资专家葛兰碧于20世纪中期提出来的。均线理论是当今应用
            最普遍的技术指标之一，它帮助交易者确认现有趋势、判断将出现的趋势等。简单移动平
            均线是最简单的一种移动平均线，它是某个时间段价格序列的简单平均值。也就是说，
            这个时间段上的每个价格权重相同
    """
    maValue = pd.DataFrame(columns = [name], index = np.arange(len(Price)))#申请内存空间
    for i in range(length, len(Price)+1):
        maValue.iloc[i-1] = Price[i - length : i].sum()/length
    maValue[0:length - 1] = pd.DataFrame(Price[0:length - 1])#Series转成表容器
    return maValue



