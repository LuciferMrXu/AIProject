#coding:utf-8

from __future__ import division
import matplotlib.pyplot as plt
# import matplotlib.finance as mpf
from matplotlib.dates import date2num #date2num
import pandas as pd
import datetime
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter



def Kplot_indictor(df):
    plt.figure(1, figsize = (15, 10), facecolor=(0.9, 0.9, 0.9))
    plt.subplot(311 ,facecolor = (0.24, 0.24, 0.24))
    Date = df.loc[:, 'time']
    kData = df.iloc[:, 1:5]#K线数据
    allData = df.iloc[:, 1:6]#所有数据
    kNum = len(df['close']) + 10#统计K线的数量，目前有18根K线
     
    isMat = kData.shape[1]#统计K先数据有多少列
    indexShift = 0
     
    if isMat == 4:
        O = kData.loc[:,'open']
        H = kData.loc[:,'high']
        L = kData.loc[:,'low']
        C = kData.loc[:,'close']
    colorDown = 'cyan' #规定线条颜色
    colorUp = 'red'
    colorLine = 'black'
    date = np.arange(len(df['close']))#K线个数
    w = 0.3#修正系数
    d = C - O#收盘价小于开盘价是阴线
    l = len(d)
#     plt.subplot(facecolor = (0.24, 0.24, 0.24))#====================================================================
 #开始画下引线
    for i in range(l):
#         plt.figure(1, figsize = (15, 10), facecolor=(0.9, 0.9, 0.9))
        if O[i] <= C[i]:#阳线下引线至上引线
            colorLine = colorUp
            plt.plot((date[i]+1-w,date[i]+1-w),  (L[i],O[i]), color = colorLine, linewidth = 0.5)
            plt.plot((date[i]+1-w,date[i]+1-w),  (C[i],H[i]), color = colorLine, linewidth = 0.5)
        if O[i] > C[i]:#阴线下引线至上引线
            colorLine = colorDown
            plt.plot((date[i]+1-w,date[i]+1-w),  (L[i],C[i]), color = colorLine, linewidth = 0.5)
            plt.plot((date[i]+1-w,date[i]+1-w),  (O[i],H[i]), color = colorLine, linewidth = 0.5)
#开始画实体
    nDown = d[d<0].index.tolist()#索引
    for j in range(len(nDown)):
#         plt.figure(1, figsize = (15, 10), facecolor=(0.9, 0.9, 0.9))
        xDown = [date[nDown[j]]+1-2*w, date[nDown[j]]+1-2*w, date[nDown[j]]+1, date[nDown[j]]+1, date[nDown[j]]+1-2*w]
        yDown = [O[nDown[j]], C[nDown[j]], C[nDown[j]], O[nDown[j]], O[nDown[j]]]
        plt.fill(xDown, yDown, colorDown)
    nUp = d[d>=0].index.tolist()#索引
    for j in range(len(nUp)):
#         plt.figure(1, figsize = (15, 10), facecolor=(0.9, 0.9, 0.9))
        xUp = [date[nUp[j]]+1-2*w, date[nUp[j]]+1-2*w, date[nUp[j]]+1, date[nUp[j]]+1, date[nUp[j]]+1-2*w]
        yUp = [O[nUp[j]], C[nUp[j]], C[nUp[j]], O[nUp[j]], O[nUp[j]]]
        plt.fill(xUp, yUp, colorUp)
       
    plt.xlim(0, kNum )
    ax = plt.gca()#获得当前作图区域句柄

    ax.xaxis.set_major_locator(MultipleLocator(kNum/(kNum/2)))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
   
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
   
    xtick, _ = plt.xticks()#获取x轴刻度
    xtick = xtick[1:-1]
    xtick = [i + 1-w for i in xtick]
    plt.xticks(xtick)
    xticklabels = []
    for i in range(int(len(Date)/2)):
        if i == 0:
            xticklabels.append(Date[i])
        else:
            xticklabels.append(Date[i*2])
   
#     ax.set_xticklabels(xticklabels, rotation = 50)
       
#     plt.show()
    return
           

