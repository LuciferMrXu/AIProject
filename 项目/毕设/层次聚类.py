#_*_ coding:utf-8_*_

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler



file=pd.read_csv('./weibo.csv')
data = file.set_index('名称', drop=True)
data_train=pd.get_dummies(data['是否认证'])
data=data.drop(columns='是否认证')
X=pd.concat([data,data_train],axis=1).astype(float)


ss = MinMaxScaler()
X = ss.fit_transform(X)


algo = AgglomerativeClustering(n_clusters=5, affinity='euclidean',linkage="ward")
algo.fit(X)

label=algo.labels_

con = pd.Series(label,index=file['名称'])
con=con.replace({3:1,4:2,1:3,0:4,2:5})
# con=con.sort_values(ascending=False)
# con=con.replace({1:'一星级',2:'两星级',3:'三星级',4:'四星级',5:'五星级'})
print(con)


result = con.value_counts()
print("=" * 50)
print(result)

