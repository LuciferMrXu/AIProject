#_*_ coding:utf-8_*_
from pymongo import MongoClient
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
# 设置中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']    # 设置字体
mpl.rcParams['axes.unicode_minus'] = False      # 中文显示

client = MongoClient('localhost', 27017)  # 连接数据库
db = client.mydb
collection1 = db.微博
collection2 = db.微博信息


data=pd.read_csv('weibo.csv')

def date_format(dt):
    import time
    if dt == '/':
        result=np.nan
    else:
        t = time.strptime(dt, "%Y.%m.%d")
        result=int(t.tm_year)
    return result



def fun1(data):
    ser=data['是否认证'].value_counts()
    print(ser)

    x=[ser[0],ser[2],ser[1]]
    # 设置颜色
    colors=['green','yellow','red']

    labels=['已认证','尚未认证','未开通']

    plt.figure()
    # 绘制饼图
    plt.pie(x,labels=labels,colors=colors,startangle=90,shadow=True,labeldistance=1.2,autopct='%.1f%%')
    # 设置为圆
    plt.axis('equal')

    plt.title('新浪微博开通情况饼状图')
    plt.show()


def fun2(data):
    data = data[['名称','创建时间']]
    date=data.iloc[:,1]
    X = date.apply(lambda x: pd.Series(date_format(x)))
    result=X[0].value_counts()
    print(result)

    height = [result[2009],result[2010],result[2011],result[2012],result[2013],result[2014],result[2015],result[2016]]

    left = np.arange(1, 9, 1)
    n = 8
    width=5/n
    plt.figure()

    plt.bar(left,height,width=width,color='lightskyblue',align='center')
    plt.plot(left,height,color='r')
    # 生成图例
    plt.legend(loc='best')

    # 设置标题
    plt.title('高校图书馆开通微博年份统计')
    # 设置x轴,y轴的标签
    plt.xlabel('年份')
    plt.ylabel('数量')

    # 设置图标的刻度
    plt.xticks(np.arange(1,9,1),['2009年','2010年','2011年','2012年','2013年','2014年','2015年','2016年'])   # 列表元素与刻度一一对应
    plt.yticks(np.arange(1, 19, 3))
    plt.show()

def fun3(data):
    data = data.replace("/", np.NAN)     # 过滤缺省值
    data = data.dropna(how='any', axis=0)
    stem_cats = ['关注','粉丝数','微博数量']
    fig = plt.figure(figsize=(8, 7))
    for sp in range(0, 3):
        ax = fig.add_subplot(1, 3, sp + 1)
        ax.boxplot(data[stem_cats[sp]])
        ax.set_title(stem_cats[sp])
        ax.set_ylabel('数量')
    plt.show()

fun3(data)