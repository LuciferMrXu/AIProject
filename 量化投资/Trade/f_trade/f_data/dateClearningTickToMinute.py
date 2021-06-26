#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import time
import datetime
import sys


def tickToOneminute(df):   
    DelStrIndex = (df['time']=='2018-03-15 14:59:00').replace(False, np.nan).dropna().index.tolist()#找到停盘时间的索引值
    pass #可加其它时间切分语句
    if len(DelStrIndex)>0:
        df.drop(df.index[DelStrIndex[0]:],inplace = True)#删除索引行以后的数据
    MM = int(60)#
    timeD = pd.DataFrame()
    timeT = pd.DataFrame()
    for i in range(len(df)):
        timeD.loc[i, 'time'] = str(df.loc[i, 'time']).split(' ')[0]
    for i in range(len(df)):
        timeT.loc[i, 'time'] = str(df.loc[i, 'time']).split(' ')[-1] 
    timeDD = timeD['time'].str.split('-')#.str[0] timeDD = timeD.str.split('/')
    timeTT = timeT['time'].str.split(':')#.str[0]
    ## 日期数值化
    time_y = pd.DataFrame(timeDD.str[0], dtype = 'int')*10000#转化为字符串型
    time_month = pd.DataFrame(timeDD.str[1], dtype = 'int')*100 #月
    time_d = pd.DataFrame(timeDD.str[2], dtype = 'int') #日
    dateN = time_y + time_month + time_d
    ## 日期数值化
    time_h = pd.DataFrame(timeTT.str[0], dtype = 'int')*100 #小时
    time_m = pd.DataFrame(timeTT.str[1], dtype = 'int') #分钟
    timeN = time_h + time_m #2018/3/15 14:07:24
    #清洁数据,适用于股票
    #timeN.replace(925, 930, inplace = True)#09:25的数据给09:30
    #timeN.replace(926, 930, inplace = True)#09:26的数据给09:30
    #timeN.replace(929, 930, inplace = True)#09:29的数据给09:30
    #timeN.replace(1130, 1129, inplace = True)#11:30的数据给11:29
    #timeN.repalce(1259, 1300, inplace = True)#12:59的数据给13:00
    #timeN.replace(1500, 1459, inplace = True)#15:00的数据给14:59
    #清洁数据，适用于期货
    timeN.replace(859, 900, inplace = True)#08:59的数据给09:00
    timeN.replace(1130, 1129, inplace = True)#11:30的数据给11:29
    timeN.replace(1329, 1330, inplace = True)#13:29的数据给13:30
    timeN.replace(1500, 1459, inplace = True)#15:00的数据给14:59
    timeN.replace(2059, 2100, inplace = True)#20:59的数据给21:00
    timeN.replace(100, 59, inplace = True)#01:00的数据给00:59,最后一条根据品种而定，LME1~19
    #数据合成，行转列思维
    u1 = timeN.drop_duplicates()#小集合
    datetimeN = dateN + timeN/10000
    u2 = datetimeN.drop_duplicates()
    # print len(u2)
    dataM1 = pd.DataFrame(columns = ['time','open','high','low','close','vol'], index = np.arange(len(u2)))
    start = time.time()#记录数据合成开始时间
    for i in range(len(u2)):
        df_ind_all = (datetimeN == u2.iloc[i]).replace(False, np.nan).dropna().index.tolist()
        ind1 = df_ind_all[0]
        ind2 = df_ind_all[-1]
        if timeN.loc[ind1, 'time'] == int(930) or (timeN.loc[ind1, 'time'] == 1300):#因为1分钟有好多个
            dataM1.loc[i,'time'] = df['time'][ind2]
        else:
            dataM1.loc[i,'time'] = df['time'][ind1]
        dataM1.loc[i, 'open'] = df.iloc[ind1, 1]
        if ind1 == ind2:
            dataM1.loc[i, 'high'] = df.iloc[ind1, 1]
            dataM1.loc[i, 'low'] = df.iloc[ind1, 1]
        else:
            dataM1.loc[i, 'high'] = max(df.iloc[ind1:ind2, 1])
            dataM1.loc[i, 'low'] = min(df.iloc[ind1:ind2, 1])
        dataM1.loc[i, 'close'] = df.iloc[ind2, 1]
        if timeN.loc[ind1, 'time'] == int(930):
            dataM1.loc[i,'vol'] = df.iloc[ind2, 2]
        else:
            dataM1.loc[i,'vol'] = df.iloc[ind2, 2] - df.iloc[ind1, 2]
#        print dataM1  
    end = time.time()#记录数据合成完成时间
#     print dataM1.head(5)
    dataM1.to_csv('./dataOneMinute.csv')#导出至csv文件
    #print('tick转换时间为%.2f,数据已保存在当前路径下的csv文件中'%(end-start))
    print('tick合成1分钟时间为%.2f'%(end-start))
    return dataM1












 
if __name__ == '__main__': 
    df = pd.read_csv('./data.csv', encoding = 'utf-8') #读入tick数据    
    DelStrIndex = (df['time']=='2018-03-15 14:59:00').replace(False, np.nan).dropna().index.tolist()#找到停盘时间的索引值
    df.drop(df.index[DelStrIndex[0]:],inplace = True)#删除索引行以后的数据
    tickToOneminute(df)





