#_*_ coding:utf-8_*_
import pandas as pd

def load(filename):
    fr=pd.read_csv(filename,sep='\t')
    print(fr.head())
    print(fr.info())
    print(fr.describe())
    returnMat=fr.iloc[:,0:38]
    classLabelVector=fr.iloc[:,-1]
    return returnMat,classLabelVector
if __name__=='__main__':
    path='../../datas/zhengqi_train.txt'
    load(path)