#_*_ coding:utf-8_*_
import pandas as pd
from preview import load
from sklearn.model_selection import train_test_split,GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error

outline_path = '../../datas/zhengqi_train.txt'
online_path = '../../datas/zhengqi_test.txt'
def clean(x,y):
    pass


def preprocession(x,y):
    x, y = load(outline_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75,test_size=0.25, random_state=16)
    return x_train, x_test, y_train, y_test


def xgboost_train(x_train, x_test, y_train, y_test):
    model=xgb.XGBRegressor(max_depth=3,
                     learning_rate=0.1,
                     n_estimators=100,
                     slice=True,   # 显示每一轮的迭代结果
                     reg_alpha=1,
                     reg_lambda=1,
                     objective='reg:linear')

    model.fit(x_train,y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    print('训练集上MSE为：%.3f\n测试集上MSE为：%.3f' % (train_mse, test_mse))
    return model

def main_sklearn():
    x,y=load(outline_path)
    clean(x,y)
    x_train, x_test, y_train, y_test=preprocession(x,y)
    xgboost_train(x_train, x_test, y_train, y_test)


if __name__=='__main__':
    main_sklearn()