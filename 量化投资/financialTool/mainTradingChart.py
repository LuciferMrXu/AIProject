#coding:utf-8
import sys
sys.path.append('..\\')
from financialTool import dateClearningTickToMinute
from financialTool import packKplot
from financialTool import ma
from financialTool import boll
from financialTool import macd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def trading_graph(df):#trading_graph
    #绘制K线图
    packKplot.Kplot_indictor(df)#工具箱中的kplot和测试中的kplot有差异
    #绘制均线
    df = df['close']
    ma_5 = ma.maIndictor(df, 5, name = 'ma_5')
    ma_10 = ma.maIndictor(df, 10, name = 'ma_10')
    plt.plot(range(len(df)), ma_5, color = 'White', linewidth = 0.4)
    plt.plot(range(len(df)), ma_10, color = 'magenta', linewidth = 0.4)
    #加载boll
    upper, middle, botton= boll.Boll_indictor(df, 20, 2, Type = 1)
    plt.plot(range(len(df)), upper, color = 'cyan', linewidth = 0.4)
    plt.plot(range(len(df)), middle, color = 'cyan', linewidth = 0.4)
    plt.plot(range(len(df)), botton, color = 'cyan', linewidth = 0.4)
    #加载macd
    plt.subplot(212, facecolor = (0.24, 0.24, 0.24))
    DIF, DEA, MACDValue = macd.macd_indictor(df, 12, 26, 9, name = 'macd')
    plt.plot(range(len(df)), DIF, color = 'white', linewidth = 0.4)
    plt.plot(range(len(df)), DEA, color = 'yellow', linewidth = 0.4)
    #画macd红绿毛
    LL = len(MACDValue)
    for i in range(LL):
        if MACDValue.iloc[i, 0]>0:#阳线下引线至上引线
            colorLine = 'r'
            plt.plot((i,i), (0,MACDValue.iloc[i]), color = colorLine, linewidth = 1)
        if MACDValue.iloc[i, 0]<=0:
            colorLine = 'cyan'
            plt.plot((i,i), (0,MACDValue.iloc[i]), color = colorLine, linewidth = 1)
    plt.plot((0, len(MACDValue)), (0, 0), color = 'gray', linewidth = 0.35)
    plt.xlim(0, len(DIF) + 10)
    plt.show()
    return

if __name__ == '__main__':
    df = pd.read_csv('../Datas/data.csv',encoding='utf-8')
    df=dateClearningTickToMinute.tickToOneminute(df)
    #print(df)
    #DelStrIndex = (df['time']=='2018-03-15 14:59:00').replace(False, np.nan).dropna().index.tolist()#找到停盘时间的索引值
    #df.drop(df.index[DelStrIndex[0]:],inplace = True)#删除索引行以后的数据
    trading_graph(df)












