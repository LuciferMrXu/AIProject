#_*_ coding:utf-8_*_
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split


offline="../../datas/ccf_offline_stage1_train.csv"
online="../../datas/ccf_online_stage1_train.csv"

#1、把线下和线上两部分数据进行合并 2、把label值变成0 和 1
def load(path1,path2):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    #做法一：把线上的action筛选后删除，合并两表格 todo 因为验证集里没有action的数据，因此选用方法一
    df2 = df2[df2['Action'] == 1].drop(['Action'],1)
    #做法二：给线下的数据添加一个action  todo 优先考虑
    #        给线上的数据添加一个Distance
    # df1['Action'] = 1

    df2['Distance'] = 0
    print(df1.shape)
    print(df2.shape)
    df = pd.concat([df1,df2],sort=False)
    # print(df.info())
    df['Date_received'] = df['Date_received'].replace("null",np.NaN)
    df = df.dropna(how='any')

    df['Date'] = pd.to_datetime(df['Date'], format="%Y%m%d")
    df['Date_received'] = pd.to_datetime(df['Date_received'], format="%Y%m%d")

    # print(df['Date'])
    # print(df['Date_received'])

    inter_list = []
    for date,date_r in zip(df['Date'],df['Date_received']):
        if date == "null":
            inter_list.append(0)
        else:
            interval = abs(date - date_r)
            if interval.days <= 15:
                inter_list.append(1)
            else:
                inter_list.append(0)
    df['label'] = inter_list
    df.drop(['Date'],1,inplace=True)
    return df

def preprocessor(df):
    print(df['Discount_rate'].value_counts())
    s1 = [] #存满100减50的100
    s2 = [] #存满100减50的50
    s3 = [] #优惠率
    for i in df['Discount_rate']:
        if i.find(":") >= 0:
            ls = i.split(":")
            s1.append(ls[0])
            s2.append(ls[1])
            s3.append(int(ls[1])/int(ls[0]))
        elif i == "fixed":
            s1.append(np.NaN)
            s2.append(np.NaN)
            s3.append(1)
        else:
            s1.append(np.NaN)
            s2.append(np.NaN)
            s3.append(1-float(i))
    df['full_price'] = s1
    df['reduction_price'] = s2
    df['reduction_rate'] = s3
    df.drop(['Discount_rate'],1,inplace=True)
    return df

df = load(offline,online)
preprocessor(df)
print(df.head())
