#_*_ coding:utf-8_*_
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler

file=pd.read_csv('./DATA/food_info.csv')
# print(file.dtypes)
Y=file['Energ_Kcal']
x1=file.iloc[:,2]
x2=file.iloc[:,4:]
X=pd.concat([x1,x2],axis=1)



x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75,test_size=0.25)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)


lgbr=lgb.LGBMRegressor(boosting_type='gbdt')
param_grid = {
    'num_leaves': [4,8,16,32],
    'learning_rate': [0.2,0.5,0.7,0.9],
    'n_estimators': [75,100,165],
    'subsample_for_bin':[800,1000,1200,1500]
}

model = GridSearchCV(estimator=lgbr, param_grid=param_grid, cv=5)

model.fit(x_train, y_train)

print("最优模型:{}".format(model.best_estimator_))
print("最优模型对应的参数:{}".format(model.best_params_))
print("训练集上的效果(准确率):{}".format(model.score(x_train, y_train)))
print("测试集上的效果(准确率):{}".format(model.score(x_test, y_test)))
