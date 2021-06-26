#coding:utf-8

from __future__ import division
from WindPy import *
import pandas as pd 
import numpy as np
import  matplotlib.pyplot as plt
import sys
from financialTool import ma
# 
# w.stop()
# w.start(waitTime = 120)

# #获取历史日线数据数据
# data = w.wsd("300085.SZ", "open,high,low,close,volume,amt", "2018-03-17", "2018-04-15", "Fill=Previous;PriceAdj=F")
# if data.ErrorCode == 0:
# #     df = pd.DataFrame(columns = [data.Fields])
#     df = pd.DataFrame()
#     df.loc[:, 'time'] = data.Times
#     df.loc[:,'open'] = data.Data[0]
#     df.loc[:,'high'] = data.Data[1]
#     df.loc[:,'low'] = data.Data[2]
#     df.loc[:,'close'] = data.Data[3]
#     df.loc[:,'volume'] = data.Data[4]
#     df.loc[:, 'amt'] = data.Data[5]
#     df.loc[:, 'code'] = data.Codes
# else:
#     print('请检查您的语法是否存在问题。。。')
##=========================================================================================    
# #获取历史分钟行情数据
# data = w.wsi("300085.SZ", "open,high,low,close,volume,amt", "2018-03-17", "2018-04-15", "Fill=Previous;PriceAdj=F")
# if data.ErrorCode == 0:
# #     df = pd.DataFrame(columns = [data.Fields])
#     df = pd.DataFrame()
#     df.loc[:, 'time'] = data.Times
#     df.loc[:,'open'] = data.Data[0]
#     df.loc[:,'high'] = data.Data[1]
#     df.loc[:,'low'] = data.Data[2]
#     df.loc[:,'close'] = data.Data[3]
#     df.loc[:,'volume'] = data.Data[4]
#     df.loc[:, 'amt'] = data.Data[5]
#     df.loc[:, 'code'] = data.Codes
# else:
#     print('请检查您的语法是否存在问题。。。')
##=========================================================================================       

# #获取日内tick行情数据
# data = w.wst("300085.SZ", "ask,volume,amt", "2018-04-16 09:00:00", "2018-04-16 14:59:00", "")
# if data.ErrorCode == 0:
# #     df = pd.DataFrame(columns = [data.Fields])
#     df = pd.DataFrame()
#     df.loc[:, 'time'] = data.Times
#     df.loc[:,'close'] = data.Data[0]
#     df.loc[:,'vol'] = data.Data[1]
#     df.loc[:, 'amt'] = data.Data[2]
#     df.loc[:, 'code'] = data.Codes
# else:
#     print('请检查您的语法是否存在问题。。。')
# df['close'].replace(0, np.nan, inplace = True)
# df['close'].fillna(method = 'ffill', inplace = True)
# print df['close']
# DelStrIndex = (df['time']=='2018-04-16 09:25:00').replace(False, np.nan).dropna().index.tolist()#找到停盘时间的索引值
# df.drop(df.index[: DelStrIndex[0]+1],inplace = True)#删除索引行以后的数据
# ##=========================================================================================    
# df_ma_5 = ma.maIndictor(df['close'], 5, name= 'ma_5')
# print df_ma_5
# plt.plot(df_ma_5)
# plt.show()
