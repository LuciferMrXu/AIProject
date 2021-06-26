#_*_ coding:utf-8_*_
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import f1_score, recall_score
pd.set_option('display.max_rows', None)

file=pd.read_csv('./DATA/classifiy/titanic_train.csv')
# print(file.dtypes)
file=file.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)


age=file['Age'].values.reshape(-1,1)
imputer = SimpleImputer(missing_values= np.NAN,strategy ="mean")
age = imputer.fit_transform(age)
embark=file['Embarked'].values.reshape(-1,1)
imputer = SimpleImputer(missing_values= np.NAN,strategy ="most_frequent")
embark = imputer.fit_transform(embark)

file['embark']=embark
x=file[['Sex','embark']]
# print(pd.isnull(file[['Sex','Embarked']]))

onehot=OneHotEncoder(categories='auto')
algo=onehot.fit_transform(x)

X=np.hstack((age,algo.toarray(),file['Pclass'].values.reshape(-1,1),file['SibSp'].values.reshape(-1,1),file['Parch'].values.reshape(-1,1),file['Fare'].values.reshape(-1,1)))
Y=file['Survived'].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75,test_size=0.25)

ss = MinMaxScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

SVM = SVC(probability=True)
param_grid = {
    'C': [ 0.1, 0.5, 1.0],
    'kernel': ['rbf'],
    'gamma': ['auto', 0.01, 0.1, 1.0],
    'degree': [2, 3]
}
model = GridSearchCV(estimator=SVM, param_grid=param_grid, cv=5)
model.fit(x_train, y_train)

print("最优模型:{}".format(model.best_estimator_))
print("最优模型对应的参数:{}".format(model.best_params_))
print("*" * 50)
print("训练集上的准确率:{}".format(model.score(x_train, y_train)))
print("测试集上的准确率:{}".format(model.score(x_test, y_test)))
print("训练集上的F1值:{}".format(f1_score(y_train, model.predict(x_train), average='macro')))
print("测试集上的F1值:{}".format(f1_score(y_test, model.predict(x_test), average='macro')))
print("训练集上的召回率:{}".format(recall_score(y_train, model.predict(x_train), average='macro')))
print("测试集上的召回率:{}".format(recall_score(y_test, model.predict(x_test), average='macro')))
