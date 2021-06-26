# -- encoding:utf-8 --
"""
Create by ibf on 2018/10/21
"""
import pandas as pd

# 1. 读取数据
file_path = './result_process02'
df = pd.read_csv(file_path)
df = df.dropna(axis=0)
print(df.label.value_counts())

