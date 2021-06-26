# -*- coding: utf-8 -*-

from f_data import dateClearningTickToMinute
import time
import sys


def tick_to_one(df):
#    global trading
    if len(df['close'])>2:#根据周期自行计算
        dfMinute = dateClearningTickToMinute.tickToOneminute(df)#tick数据合成1分钟数据
        dfMinute.dropna(axis = 0, how = 'any', inplace = True)
        if len(dfMinute) > 7:
            dfMinute = dfMinute.iloc[-7:, :]
        print dfMinute
        return dfMinute
    else:
        print u'数据量不够请等待获取。。。。。。。。。。。。。。。。。。。'
