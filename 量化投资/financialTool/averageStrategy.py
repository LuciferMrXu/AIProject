#coding:utf-8

from __future__ import division
import pandas as pd
import numpy as np
from financialTool import ma
from financialTool import dateClearningTickToMinute
from financialTool import tradingChart
import sys
import matplotlib.pyplot as plt

df = pd.read_csv('../data/data.csv')#实盘时可加载tick数据
df = dateClearningTickToMinute.tickToOneminute(df)
open = df['open']
close = df['close']
ma_5 = ma.maIndictor(close, 5, name='ma_5')
ma_10 = ma.maIndictor(close, 10, name='ma_10')
tradingChart.trading_graph(df)
initialCapital = pd.DataFrame(columns = ['banlance'], index = range(len(df)))
initialCapital.fillna(10000, inplace = True)
pos = pd.DataFrame(columns = ['pos'], index = range(len(df)))
pos.fillna(0, inplace = True)
# buy_pos = 0
# short_pos = 0
for i in range(10, len(close)):
    buy_condition = ma_5.loc[i-1, 'ma_5'] > ma_10.loc[i-1, 'ma_10']
    sell_condition =  ma_5.loc[i-1, 'ma_5'] < ma_10.loc[i-1, 'ma_10']
    short_condition = ma_5.loc[i-1, 'ma_5'] < ma_10.loc[i-1, 'ma_10']
    cover_condition = ma_5.loc[i-1, 'ma_5'] > ma_10.loc[i-1, 'ma_10']
    
    if ((buy_condition) and (pos.loc[i-1, 'pos'] == 0)):#and buy_pos == 0 and short_pos == 0:
        pass#加载接口信号
        plt.subplot(311, facecolor = (0.24, 0.24, 0.24))
        plt.plot(i, open[i], '*', color = 'red' )
        plt.text(i, open[i], 'buy', color = 'red')
        pos.loc[i, 'pos'] = 1
#         buy_pos = 1
        initialCapital.loc[i, 'banlance'] = initialCapital.loc[i-1, 'banlance'] + close[i]-open[i]
        continue#防止在同一根K线卖出
    if ((sell_condition) and (pos.loc[i-1, 'pos'] == 1)):# and buy_pos ==1:
        pass#加载接口信号
        plt.subplot(311, facecolor = (0.24, 0.24, 0.24))
        plt.plot(i, open[i], 'ro', color = 'yellow' )
        plt.text(i, open[i], 'sell', color = 'yellow')
        pos.loc[i, 'pos'] = 0
#         buy_pos = 0
        initialCapital.loc[i, 'banlance'] = initialCapital.loc[i-1, 'banlance'] + open[i] - close[i-1]
        continue
    if ((short_condition) and (pos.loc[i-1, 'pos'] == 0)):# and buy_pos == 0 and short_pos == 0:
        pass#加载接口信号
        plt.subplot(311, facecolor = (0.24, 0.24, 0.24))
        plt.plot(i, open[i], '*', color = 'cyan' )
        plt.text(i, open[i], 'short', color = 'cyan')
        pos.loc[i, 'pos'] = -1
#         short_pos = 1
        initialCapital.loc[i, 'banlance'] = initialCapital.loc[i-1, 'banlance'] + open[i] - close[i]
        continue
    if ((cover_condition) and (pos.loc[i-1, 'pos'] == -1)):# and short_pos == 1:
        pass#加载接口信号
        plt.subplot(311, facecolor = (0.24, 0.24, 0.24))
        plt.plot(i, open[i], 'ro', color = 'yellow' )
        plt.text(i, open[i], 'cover', color = 'yellow')
        pos.loc[i, 'pos'] = 0
#         short_pos = 0
        initialCapital.loc[i, 'banlance'] = initialCapital.loc[i-1, 'banlance'] + close[i-1] - open[i]
        continue
    #仓位管理很重要
    if pos.loc[i-1, 'pos'] == 1:
        pos.loc[i, 'pos'] = 1
        initialCapital.loc[i, 'banlance'] = initialCapital.loc[i-1, 'banlance'] + close[i] - close[i-1]
    if pos.loc[i-1, 'pos'] == -1:
        pos.loc[i, 'pos'] = -1
        initialCapital.loc[i, 'banlance'] = initialCapital.loc[i-1, 'banlance'] + close[i-1] - close[i]
    if pos.loc[i-1, 'pos'] == 0:
        pos.loc[i, 'pos'] = 0
        initialCapital.loc[i, 'banlance'] = initialCapital.loc[i-1, 'banlance']
#加载资金曲线
plt.subplot(313, facecolor = (0.24, 0.24, 0.24))
plt.plot(range(len(df)), initialCapital, color = 'magenta', linewidth = 0.4 )
plt.xlim(0, len(df)+10)
plt.plot(len(df), 10000,  label = 'capital', color ='magenta')
plt.legend()


plt.show()