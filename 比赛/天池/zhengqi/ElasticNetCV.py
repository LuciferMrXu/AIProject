#_*_ coding:utf-8_*_
import numpy as np
import pandas as pd
from preview import load
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def train_test(path,random_state=16):
    x,y=load(path)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.8,random_state=random_state)
    return x_train,x_test,y_train,y_test

def train_model(path,random_state=16):
    x_train, x_test, y_train, y_test=train_test(path)
    pipeline = Pipeline(steps=[
        ('ss',StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('algo', ElasticNet(random_state=random_state))
    ])
    params = {
        "poly__degree": [1,2,3,4,5],
        # "algo__alpha": [0.1,0.01,0.5,0.75,1],
        "algo__l1_ratio": [0.5,0.75],
        "algo__fit_intercept": [True, False]
    }
    algo = GridSearchCV(estimator=pipeline, cv=3, param_grid=params)
    algo.fit(x_train, y_train)
    print("最优参数:{}".format(algo.best_params_))
    y_pred_train=algo.predict(x_train)
    y_pred_test = algo.predict(x_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse=mean_squared_error(y_test,y_pred_test)
    print('训练集上MSE为：%.3f\n测试集上MSE为：%.3f'%(train_mse,test_mse))
    return algo

def online_predict(model,online,online_outpath):
    online_test=pd.read_csv(online,sep='\t')
    online_pred=model.predict(online_test)
    with open(online_outpath,'w') as f:
        online_list=[str(i) for i in online_pred]
        f.write('\n'.join(online_list))
    
    
if __name__=='__main__':
    train_path='../../datas/zhengqi_train.txt'
    online_path='../../datas/zhengqi_test.txt'
    online_outpath='../../results/zhengqi.txt'
    modle=train_model(train_path)
    online_predict(modle,online_path,online_outpath)