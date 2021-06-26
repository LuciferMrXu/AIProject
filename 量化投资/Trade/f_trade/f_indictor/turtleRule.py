#coding:utf-8

from __future__ import division
import pandas as pd
import numpy as np
import tushare as ts
import sys
import matplotlib.pyplot as plt

df = ts.get_hist_data('300085')
#定义参数
N1 = 20
N2 = 10
#数据处理
df = df.sort_index()#按照索引升序排列

df['最近N1个交易日最高价'] = df['high'].rolling(N1).max()#pd.rolling_max(df.high, N1)#查找近N1个交易日最高价
df['最近N1个交易日最高价'].fillna(value = df['high'].expanding().max(), inplace = True)#pd.expanding_max(df['high']), inplace = True)#用最高价填充空值
df['最近N2个交易日最低价'] = df['low'].rolling(N2).min()
df['最近N2个交易日最低价'].fillna(value = df['low'].expanding().min(), inplace = True)
#计算买入信号/平空头条件
buy_index = df[df['close'] > df['最近N1个交易日最高价'].shift(1)].index
df.loc[buy_index, '收盘发出的信号'] = 1
#计算卖出信号/平多头条件
sell_index = df[df['close'] < df['最近N2个交易日最低价'].shift(1)].index
df.loc[sell_index, '收盘发出的信号'] = 0#期货的话改成-1即可
#计算仓位
df['当天的仓位'] = df['收盘发出的信号'].shift(1)
df['当天的仓位'].fillna(method = 'ffill', inplace = True)
#截取数据
df = df.truncate(after = '2016-06-16')
#计算收益
df['net'] = (df['p_change']/100 * df['当天的仓位'] + 1.0).cumprod()#累乘
#设置初始资金值
initial_idx = df.iloc[0]['close']/(1 + df.iloc[0]['p_change']/100)#前一天收盘价
df['capital_index'] = df['net'] * initial_idx #资金指数
#计算策略收益率
df['策略每日涨跌幅'] = df['p_change'] * df['当天的仓位']
##计算年华收益率
year_rtn = df['策略每日涨跌幅'].cumsum()[-1]/len(df)*273
print '策略年华收益率为%.2f%%'%year_rtn
#输出至制定文件
df.drop(['price_change','ma5','ma10','ma20','v_ma5','v_ma10','v_ma20','turnover','收盘发出的信号'],axis = 1, inplace = True)
df.to_csv('./turtleRuleRevenue.csv')
plt.figure(1, figsize = (15, 10), facecolor=(0.9, 0.9, 0.9))
plt.subplot(211, facecolor = (0.24, 0.24, 0.24))
plt.xlim(0, len(df) + 1000)
plt.plot(range(len(df)), df['capital_index'], color = 'magenta', linewidth = 1)
plt.subplot(212, facecolor = (0.24, 0.24, 0.24))
plt.plot(range(len(df)), df['net'], color = 'yellow', linewidth = 1)
plt.xlim(0, len(df) + 1000)

plt.show()

# df.to_csv('./dataClose.csv')

