#_*_ coding:utf-8_*_
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

file=pd.read_csv('./DATA/classifiy/fandango_score_comparison.csv')
value = file.set_index('FILM', drop=True)
Year=[]
for i in value.index:
    if i.endswith('(2014)'):
        Year.append(2014)
    else:
        Year.append(2015)
value['Year']=Year
# print(value)

ss = MinMaxScaler()
x = ss.fit_transform(value)

algo = SpectralClustering(n_clusters=5, n_init=10,affinity='nearest_neighbors',n_neighbors=7)
algo.fit(x)

label=algo.labels_

con = pd.Series(label,index=file['FILM'])
result = con.value_counts()

print(con)
print(result)


# 轮廓系数越小说明样本点越离散，越大说明样本点越聚集
print('轮廓系数为：',silhouette_score(x,label))

