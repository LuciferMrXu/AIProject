#_*_ coding:utf-8_*_
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor


file=pd.read_csv('./DATA/regress/world_alcohol.txt')
file = file.dropna(how='any', axis=0)
# print(file['WHO region'].value_counts())
Y=file.iloc[:,-1].values
age=file.iloc[:,0].values.reshape(-1,1)
# print(label)
x=file.iloc[:,1:4]
# print(x)
onehot=OneHotEncoder(categories='auto')
algo=onehot.fit_transform(x)
# print(algo.toarray().shape)
X=np.hstack((age,algo.toarray()))
# print(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75,test_size=0.25)

ss = MinMaxScaler()
x_train = ss.fit_transform(x_train)
x_test=ss.transform(x_test)

GBDT= GradientBoostingRegressor()
param_grid = {
    'n_estimators': [30,50,70,100],
    'learning_rate': [0.2,0.5,0.7,1],
    'max_depth': [2,3,5],
    'alpha':[0.3,0.5,0.7,0.9]
}
model = GridSearchCV(estimator=GBDT, param_grid=param_grid, cv=5)

model.fit(x_train, y_train)

print("最优模型:{}".format(model.best_estimator_))
print("最优模型对应的参数:{}".format(model.best_params_))
print("训练集上的效果(准确率):{}".format(model.score(x_train, y_train)))
print("测试集上的效果(准确率):{}".format(model.score(x_test, y_test)))



