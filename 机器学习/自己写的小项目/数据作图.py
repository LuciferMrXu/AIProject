#_*_ coding:utf-8_*_
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_excel('./DATA/data1.csv')
data=data.T
# print(data)
col=data.loc['Region']
col[0]='pollutant'
# print(col)
data=data.iloc[1:]
data.columns=col
data=data.reset_index()
data=data.replace({'Unnamed: 2':'2005-2007','Unnamed: 3':'2005-2007',
                    'Unnamed: 5':'2007-2009','Unnamed: 6':'2007-2009',
                    'Unnamed: 8':'2009-2011','Unnamed: 9':'2009-2011',
                    'Unnamed: 11': '2011-2013', 'Unnamed: 12': '2011-2013',
                    'Unnamed: 14': '2013-2015', 'Unnamed: 15': '2013-2015',
                   })
data=data.set_index(['index','pollutant'])
print(data)
data1=data.loc['2005-2007']
print(data1)

fig=plt.figure()
data.plot(kind='bar')
plt.show()