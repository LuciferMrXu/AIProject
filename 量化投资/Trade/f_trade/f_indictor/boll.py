#coding:utf-8
from __future__ import division
import pandas as pd
import numpy as np
from financialTool.ma import maIndictor#导入均线包
from financialTool.ema import ema_indictor



def Boll_indictor(Price, Length, Width, Type = 1):#**kwargs
    """此函数可以计算布林轨线，编写者：Aellen
        BOLL指标又叫布林线指标，其英文全称是“Bolinger Bands”,是用该指标的创立人约翰·布林 的姓来命名的，
        是研判价格运动趋势的一种中长期技术分析工具。BOLL指标是美国股市分析家约翰·布林根据统计学中的标准差原理设计
        出来的一种非常简单实用的技术分析指标。一般而言，价格的运动总是围绕某一价值中枢（如均线、成本线等）在一定的
        范围内变动，布林线指标指标正是在上述条件的基础上，引进了“价格通道”的概念，其认为价格通道的宽窄随着价格波动
        幅度的大小而变化，而且价格通道又具有变异性，它会随着价格的变化而自动调整。正是由于它具有灵活性、直观性和趋
        势性的特点，BOLL指标渐渐成为投资者广为应用的市场上热门指标。BOLL 是利用“价格通道”来显示价格的各种价位，
        当价格波动很小，处于盘整时，价格通道就会变窄，这可能预示着价格的波动处于暂时的平静期；当价格波动超出狭窄的
        价格通道的上轨时，预示着价格的异常激烈的向上波动即将开始；当价格波动超出狭窄的价格通道的下轨时，同样也预示
        着价格的异常激烈的向下波动将开始。
    """
    ML = pd.DataFrame(columns = ['middle'], index = np.arange(len(Price)))#申请内存空间
    UL = pd.DataFrame(columns = ['upper'], index = np.arange(len(Price)))#申请内存空间
    LL = pd.DataFrame(columns = ['lower'], index = np.arange(len(Price)))#申请内存空间
    if Type == 1:
        ML = maIndictor(Price, Length, name = 'MA')
        UL[:Length - 1]= ML[:Length - 1]
        LL[:Length - 1]= ML[:Length - 1]
        for i in range(Length, len(Price)+1):
            UL.iloc[i - 1] = float(ML.iloc[i - 1] + Width*np.std(Price[i - Length : i]))
            LL.iloc[i - 1] = float(ML.iloc[i - 1] - Width*np.std(Price[i - Length : i]))
    if Type == 2:
        ML = ema_indictor(Price, Length, name = 'middle')
        UL[:Length - 1]= ML[:Length - 1]
        LL[:Length - 1]= ML[:Length - 1]
#         StanDev = pd.DataFrame(columns = ['stand'], index = np.arange(len(Price)))#申请内存空间
        for i in range(Length, len(Price)+1):
            stand = np.sqrt(((np.array(Price.iloc[i-Length:i])-np.array(ML.iloc[i-1]))**2).sum()/Length)
            UL.iloc[i - 1] = float(ML.iloc[i - 1]) + Width*stand
            LL.iloc[i - 1] = float(ML.iloc[i - 1]) - Width*stand
#     print UL, ML, LL   
    return UL, ML, LL
        

if __name__ == '__main__':
    df = pd.read_csv('../data/dataOneMinute.csv', encoding = 'utf-8')#加载数据源================
    DelStrIndex = (df['time']=='2018-03-15 14:59:00').replace(False, np.nan).dropna().index.tolist()#找到停盘时间的索引值
    df.drop(df.index[DelStrIndex[0]:],inplace = True)#删除索引行以后的数据
    df = df['close'] #清理后数据
    Length = 20 #均线周期
    Width = 2#步林轨线宽度
    i = 21
    Boll_indictor(df, Length, Width, Type = 1)
    

    