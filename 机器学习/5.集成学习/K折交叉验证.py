#_*_ coding:utf-8_*_
from sklearn.model_selection import KFold
import numpy as np

a=np.zeros((10,10))
kf=KFold(n_splits=5,shuffle=False)  # 划分成5份
for train_index,test_index in kf.split(a):
    print('train',train_index)  # train占4份
    print('test',test_index)    # test占1份（不管几折都只占1份）