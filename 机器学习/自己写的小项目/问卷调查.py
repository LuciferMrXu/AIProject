#_*_ coding:utf-8_*_
import pandas as pd
pd.set_option('display.max_columns', None)


file=pd.read_csv('./DATA/thanksgiving-2015-poll-data.csv',encoding='gbk')

print(file.head())