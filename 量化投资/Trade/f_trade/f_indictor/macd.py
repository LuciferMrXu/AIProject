#coding:utf-8
from __future__ import division
import pandas as pd
import numpy as np
from financialTool.ema import ema_indictor



def macd_indictor(Price, FastLength, SlowLength, DEALength, name):
    """MACD技术指标，编者：Allen
       MACD（Moving Average Convergence Divergence）称为指数平滑异，
          由杰拉尔德·阿佩尔(GeraldcAppel)创造.同平均线均线，是技术分析领域应用广泛的指标之一，
         包含了三个参数， 常用的设置有（12，26，9）。MACD是计算两条不同速度（长期与中期）的指数
         平滑移动平均线（EMA）的差离状况来作为研判行情的基础。DIF为12周期均值与26周期均值之差，
    DEA为DIF的9周期均值，而MACD则为DIF与DEA差值的两倍。
    """
    MACDValue = pd.DataFrame(columns = [name], index = np.arange(len(Price)))#申请内存空间
    DIF = ema_indictor(Price, FastLength, name = 'fastEma') - ema_indictor(Price, SlowLength, name = 'fastEma')
    DEA = ema_indictor(DIF, DEALength, name = 'fastEma')
    MACDValue = (DIF-DEA)*2
    return DIF, DEA, MACDValue



























if __name__ == '__main__':
    df = pd.read_csv('./dataOneMinute.csv')
    DelStrIndex = (df['time']=='2018-03-15 14:59:00').replace(False, np.nan).dropna().index.tolist()#找到停盘时间的索引值
    df.drop(df.index[DelStrIndex[0]:],inplace = True)#删除索引行以后的数据
    Price = df['close']
    FastLength = 12
    SlowLength = 26
    DEALength = 9
    name = 'macd'
    DIF, DEA, MACDValue = macd_indictor(Price, FastLength, SlowLength, DEALength, name)
    print DIF, DEA, MACDValue
    