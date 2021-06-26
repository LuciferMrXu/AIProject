#_*_ coding:utf-8_*_
from pymongo import MongoClient
import numpy as np
import pandas as pd
# pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
mpl.rcParams['font.sans-serif'] = ['SimHei']    # 设置字体
mpl.rcParams['axes.unicode_minus'] = False      # 中文显示


client = MongoClient('localhost', 27017)  # 连接数据库
db = client.mydb
collection1 = db.微博
collection2 = db.微博信息


def choose(data1,data2,school='同济大学图书馆'):
    data1 = pd.DataFrame(list(data1.find()))   # 合并两张表
    data1 = data1.drop(['_id', '篇数'], axis=1)
    data2 = pd.DataFrame(list(data2.find()))
    data2 = data2.drop(['_id'], axis=1)
    data2 = data2.rename(columns={'关注': '总关注'})

    data=pd.merge(data1,data2,on='高校',how='inner')  # 生成层次索引
    new_date=data.set_index(['高校','粉丝','总关注','是否认证','介绍'])

    df=new_date.loc[school][['发表时间','关注','评论','转发','摘要','来源']].reset_index()  # 提取具体哪所高校的数据

    X = df[['发表时间','关注','评论','转发','摘要','来源']]

    return X

# df=choose(collection1,collection2)
#
# df1=df.sort_values(by='关注',ascending=False)
# print(df1[['发表时间','关注']])
# print('=======================')
# df2=df.sort_values(by='评论',ascending=False)
# print(df2[['发表时间','评论']])
# print('=======================')
# df3=df.sort_values(by='转发',ascending=False)
# print(df3[['发表时间','转发']])
# print('=======================')
# print(df['摘要'].count())
# print(df['摘要'].value_counts())

def date_format(dt):
    import time
    try:
        t = time.strptime(dt, "%Y-%m-%d")
    except Exception:
        X = 2018
    else:
        X = int(t.tm_year)
    return X


def test(data1,data2):
    value=choose(data1,data2)
    test1=value['来源'].value_counts()
    date=value['发表时间']
    X = date.apply(lambda x: pd.Series(date_format(x)))
    test2=X[0].value_counts().sort_index(ascending=False)
    test3=value['摘要'].value_counts()
    print(value['摘要'].count())
    print(value['摘要'].value_counts())


    pattern = re.compile('#(?!\s*#)[^#]+#')
    index=['原创']
    for i in test3.index:
        tips=pattern.findall(i)
        if tips!=[]:
            index=index+tips
    index=set(index)
    n=0
    for j in index:
        try:
            n+=test3[j]
        except Exception:
            pass
    ser = pd.Series((n,value['摘要'].count()-n), index=['原创微博', '转发微博'])

    plt.figure()

    a1 = test1[0]
    a2 = test1[1]
    a3 = test1[2]
    a4 = test1[3]
    a5 = test1[4]
    a6 = test1[5]
    a7 = test1.sum()-a1-a2-a3-a4-a5-a6
    x=[a1,a2,a3,a4,a5,a6,a7]
    labels=[test1.index[0],test1.index[1],test1.index[2],test1.index[3],test1.index[4],test1.index[5],'其他']
    plt.subplot(223)
    plt.pie(x,labels=labels,autopct='%.1f%%')
    plt.subplot(211)
    test2.plot()
    plt.title('重庆大学图书馆官方微博发布情况统计')
    plt.subplot(224)
    plt.pie(ser.values,labels=ser.index,autopct='%.1f%%')

    plt.show()




test(collection1,collection2)



