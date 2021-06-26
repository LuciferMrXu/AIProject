#_*_ coding:utf-8_*_
from pymongo import MongoClient
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
mpl.rcParams['font.sans-serif'] = ['SimHei']    # 设置字体
mpl.rcParams['axes.unicode_minus'] = False      # 中文显示
import json
import re
import math
client = MongoClient('localhost', 27017)  # 连接数据库
db = client.mydb
collection1 = db.微博
collection2 = db.微博信息


def choose(data1,data2,school='厦大图书馆'):
    data1 = pd.DataFrame(list(data1.find()))   # 合并两张表
    data1 = data1.drop(['_id', '篇数'], axis=1)
    data2 = pd.DataFrame(list(data2.find()))
    data2 = data2.drop(['_id'], axis=1)
    data2 = data2.rename(columns={'关注': '总关注'})

    data=pd.merge(data1,data2,on='高校',how='inner')  # 生成层次索引
    new_date=data.set_index(['高校','粉丝','总关注','是否认证','介绍'])

    df=new_date.loc[school][['发表时间','关注','评论','转发','摘要','正文']].reset_index()  # 提取具体哪所高校的数据

    X = df[['发表时间','关注','评论','转发','摘要','正文']]

    return X


def index(data1,data2):
    value = choose(data1, data2)
    value['h指数']=1/3*(value['关注']+value['评论']+value['转发'])
    value=value.drop(['发表时间', '关注','评论','转发'], axis=1)
    new_value=value.sort_values(by='h指数',ascending=False)
    new_value = new_value.reset_index(drop=True)
    n = 0
    for i in new_value['h指数']:
        if i>=new_value.index.tolist()[n]:
            n+=1
    new_value=new_value[0:n]
    print(new_value)
    db = client.mydb
    collection = db.厦门大学h指标
    collection.insert(json.loads(new_value.T.to_json()).values())


def date_format(dt):
    import time
    try:
        t = time.strptime(dt, "%Y-%m-%d")
    except Exception:
        X = 2018
    else:
        X = int(t.tm_year)
    return X

def BCI(data1,data2,time=2014):
    value = choose(data1, data2)
    value['发表时间'] = value['发表时间'].apply(lambda x: pd.Series(date_format(x)))
    value = value.drop(['正文'], axis=1)
    date=dict(list(value.groupby('发表时间')))[time]
    df=date[['关注','评论','转发']].cumsum()
    zan=df.iloc[-1,0]
    pin=df.iloc[-1,1]
    zhuan = df.iloc[-1, 2]
    num=date['发表时间'].count()
    test=date['摘要'].value_counts()

    pattern = re.compile('#(?!\s*#)[^#]+#')
    index=['原创']
    for i in test.index:
        tips=pattern.findall(i)
        if tips!=[]:
            index=index+tips
    index=set(index)

    date = date.reset_index(drop=True)

    n=0
    for j in date['摘要']:
        if j not in index:
            date=date.drop(n)
        n += 1

    df0 = date[['评论', '转发']].cumsum()
    self_pin=df0.iloc[-1,0]
    self_zhuan = df0.iloc[-1, 1]
    self_num=date['发表时间'].count()

    w1=0.3*math.log(num+1,math.e)+0.7*math.log(self_num+1,math.e)
    w2=0.2*math.log(zhuan+1,math.e)+0.2*math.log(pin+1,math.e)+\
            0.25 * math.log(self_zhuan + 1, math.e) + 0.2*math.log(self_pin+1,math.e)+\
            0.1 * math.log(zan + 1, math.e)
    BCI=(0.2*w1+0.8*w2)*160
    return BCI


if __name__=='__main__':
    index(collection1,collection2)

    time=[2012,2013,2014,2015,2016,2017,2018]
    value=[]
    for i in time:
        value.append(BCI(collection1,collection2,i))
    print(value)
    height = value
    left = np.arange(1, 8, 1)
    n = 8
    width=5/n
    plt.figure()

    plt.bar(left,height,width=width,color='lightskyblue',align='center')
    plt.plot(left,height,color='r')
    # 生成图例
    plt.legend(loc='best')

    # 设置标题
    plt.title('厦门大学图书馆官方微博年度BCI指标趋势图')
    # 设置x轴,y轴的标签
    plt.xlabel('年份')
    plt.ylabel('BCI')

    # 设置图标的刻度
    plt.xticks(np.arange(1,8,1),time)   # 列表元素与刻度一一对应
    plt.yticks(np.arange(0, 1650, 200))
    plt.show()