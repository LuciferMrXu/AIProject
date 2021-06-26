#coding:utf-8
from __future__ import division
import pandas as pd
import numpy as np


def ema_indictor(Price, Length, name):
    """
    EMA指标，编者：Aellen
        移动平均线是由著名的美国投资专家葛兰碧于20世纪中期提出来的。均线理论是当今应用
        最普遍的技术指标之一，它帮助交易者确认现有趋势、判断将出现的趋势等。指数移动平
        均线是比较流行的一种移动平均线，它认为较近的价格更有价值。也就是说，它赋予较近
        的价格更大的权重
    """
    EMAvalue= pd.DataFrame(columns = [name], index = np.arange(len(Price)))#申请内存空间
    K = 2/(Length + 1)
    for i in range(len(Price)):
        if i == 0:
            EMAvalue.iloc[i] = Price.iloc[0]
        else:
            EMAvalue.iloc[i] = float(Price.iloc[i]*K + EMAvalue.iloc[i-1]*(1-K))
    return EMAvalue






















if __name__ == '__main__':
    #程序参数
    df = pd.read_csv('./dataOneMinute.csv')
    DelStrIndex = (df['time']=='2018-03-15 14:59:00').replace(False, np.nan).dropna().index.tolist()#找到停盘时间的索引值
    df.drop(df.index[DelStrIndex[0]:],inplace = True)#删除索引行以后的数据
    Price = df['close']
    Length = 12
    EMAvalue =  ema_indictor(Price, Length, name = 'ema')
    print(EMAvalue)




    