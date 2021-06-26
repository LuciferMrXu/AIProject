#coding:utf-8


from __future__ import division
import tushare as ts
import pandas as pd

#for loop:
df = ts.get_k_data('sh')

h5 = pd.HDFStore('../data/test_s.h5', 'w')
h5['data']= df
h5.close()
# h5['data'] = b
df_read = pd.read_hdf('../data/test_s.h5')
print df_read