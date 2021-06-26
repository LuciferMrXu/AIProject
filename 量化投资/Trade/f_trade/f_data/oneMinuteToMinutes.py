#coding:utf-8

from __future__ import division#一般要写在最开头
import pandas as pd
import numpy as np
import time
import sys

df = pd.read_csv('./dataOneMinute.csv', encoding = 'utf-8')#读入1分钟数据
df.drop(df.columns[0], axis = 1, inplace = True)#删除原来DataFrame编码，边写边调试
DelStrIndex = (df['time']=='2018-03-15 14:59:00').replace(False, np.nan).dropna().index.tolist()#找到停盘时间的索引值
df.drop(df.index[DelStrIndex[0]:],inplace = True)#删除索引行以后的数据  2018-03-15 14:07:24
MM = 3#MM值初始化
#开始数据切分
timeD = df['time'].str.split(' ').str[0]#年月日，转换为字符串 2018-03-15
timeT = df['time'].str.split(' ').str[-1]#时分秒，转换为字符串14:07:24
timeDD = timeD.str.split('-')
timeTT = timeT.str.split(':')
#年月日切分
time_y = pd.DataFrame(timeDD.str[0], dtype = 'int')#2018-03-15
time_month = pd.DataFrame(timeDD.str[1], dtype = 'int')
time_d = pd.DataFrame(timeDD.str[2], dtype = 'int')
#时分秒切分
time_h = pd.DataFrame(timeTT.str[0], dtype = 'int')#14:07:24
time_m = pd.DataFrame(timeTT.str[1], dtype = 'int')
timeHM = time_h*100 + time_m#== 1407
# 区分区间合成
if MM == 60:
    timeHM1 = pd.DataFrame((930<= np.array(timeHM)) & (np.array(timeHM) < 1030), columns = ['time'])
    timeHM1.time.replace(True, 1, inplace = True)#
    timeHM1.time.replace(False, np.nan, inplace = True)
    timeHM1.dropna(axis = 0, how = 'any', inplace = True )
        
    timeHM2 = pd.DataFrame((1030<= np.array(timeHM)) & (np.array(timeHM) < 1130), columns = ['time'])
    timeHM2.time.replace(True, 2, inplace = True)#
    timeHM2.time.replace(False, np.nan, inplace = True)
    timeHM2.dropna(axis = 0, how = 'any', inplace = True )
        
    timeHM3 = pd.DataFrame((1330<= np.array(timeHM)) & (np.array(timeHM) < 1430), columns = ['time'])
    timeHM3.time.replace(True, 3, inplace = True)#
    timeHM3.time.replace(False, np.nan, inplace = True)
    timeHM3.dropna(axis = 0, how = 'any', inplace = True )
        
    timeHM4 = pd.DataFrame((1430<= np.array(timeHM)) & (np.array(timeHM) < 1500), columns = ['time'])
    timeHM4.time.replace(True, 4, inplace = True)#
    timeHM4.time.replace(False, np.nan, inplace = True)
    timeHM4.dropna(axis = 0, how = 'any', inplace = True )
    timeHM_end = timeHM1.append([timeHM2, timeHM3, timeHM4])
elif MM == 3 or MM == 5 or MM == 10 or MM == 15 or MM == 30:
    time_m = time_m//MM#向下取整
    timeHM_end = time_h*100 + time_m
else:
    print '输入MM参数错误，请重新输入_____'
    sys.exit(0)
#总日期输出
datetimeN = (time_y*10000 + time_month*100 + time_d) + (timeHM_end/10000)#20180315.0001  20180315.0002  20180315.0003   20180315.0004
u = datetimeN.drop_duplicates()#做了一个unique
dataMN = pd.DataFrame(columns = ['time','open','high','low','close','vol'], index = np.arange(len(u)))#初始化申请内存空间
start = time.time()#记录数据合成开始时间   
for i in range(len(u)):
    df_ind_all = (datetimeN == u.iloc[i]).replace(False, np.nan).dropna().index.tolist()
    ind1 = df_ind_all[0]
    ind2 = df_ind_all[-1]
    dataMN.loc[i,'time'] = df['time'][ind1]#1分钟只有一个
    dataMN.loc[i, 'open'] = df.iloc[ind1, 1]
    dataMN.loc[i, 'high'] = max(df.iloc[ind1:ind2, 2])
    dataMN.loc[i, 'low'] = min(df.iloc[ind1:ind2, 3])
    dataMN.loc[i, 'close'] = df.iloc[ind2, 4]
    dataMN.loc[i,'vol'] = sum(df.iloc[ind1:ind2, 5])#1分钟只有1个
end = time.time()#记录数据合成完成时间
# print dataMN.head(60)
print(end-start)
dataMN.to_csv('./OneToMinuts.csv')#导出至csv文件
