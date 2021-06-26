#_*_ coding:utf-8_*_
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np



path = './DATA/regress/UNRATE.csv'
df = pd.read_csv(filepath_or_buffer=path,sep=',')
# print(df.head())


def date_format(dt):
    import time
    t = time.strptime(dt, "%Y-%m-%d")
    return (t.tm_year, t.tm_mon, t.tm_mday)


date = df.iloc[:,0]

X = date.apply(lambda x: pd.Series(date_format(x)))
X.iloc[:,-1] = df.iloc[:,-1]
print(X)



model = DBSCAN(eps=1.2, min_samples=3)
model.fit(X)

# 直接获取X对应的所属类别标签(返回-1表示未知的类别，不能对当前样本产生预测值)
y_hat = model.labels_

unique_y_hat = np.unique(y_hat)
n_clusters = len(unique_y_hat) - (1 if -1 in y_hat else 0)
print("类别:", unique_y_hat, "；聚类簇数目:", n_clusters)