#_*_ coding:utf-8_*_

import pandas as pd
import numpy as np


# def MaxMinNormalization(x,Max,Min):
# 	x = (x - Min) / (Max - Min)
# 	return x


a=pd.DataFrame([[1,2,3,4,5,6,7],[11,12,13,14,15,16,20]])
print(a)

a=a.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)),axis=1)
print(a)


# b=pd.Series([1,2,3,4,5,6,7])
# cor=a.corr(b)
# print(cor)





