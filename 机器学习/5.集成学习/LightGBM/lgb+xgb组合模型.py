import xgboost as xgb
import  pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
#todo xgboost有两种不同的操作形式 1、原生形式 2、sklearn形式


#todo 数据读取函数
outline_path = "../datas/zhengqi_train.txt"
online_path = "../datas/zhengqi_test.txt"
def load_pandas(path):
    df = pd.read_csv("../datas/zhengqi_train.txt",sep="\t")
    x = df.drop(['target'],1)
    y = df['target']
    return x,y
#todo 如果数据是libsvm格式的话可以使用xgboost直接读取
# TODO 额外一点，xgboost原生的优点 可以处理libsvm格式

def clean(x,y):
    return 0
#todo 特征工程
def preprocession(x,y):
    #继续实验 把特征重要性比较低的特征删掉之后，poly会不会在同一套参数下超越不加poly
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=123)
    x_train = x_train.drop(['V13' ,'V26', 'V34', 'V35', 'V36'],axis=1)
    x_test = x_test.drop(['V13' ,'V26', 'V34', 'V35', 'V36'],axis=1)
    poly = PolynomialFeatures(degree=2,include_bias=False,interaction_only=True)
    x_train = poly.fit_transform(x_train)
    x_test = poly.transform(x_test)
    return x_train,x_test,y_train,y_test

#lgb的模型
def lgb_tarin(x_train,x_test,y_train,y_test):
    model = lgb.LGBMRegressor(max_depth=3,
                     learning_rate=0.05,
                     n_estimators=100,
                     silent=True,
                     reg_alpha=1,
                     reg_lambda=1)
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    print("train mse:", mean_squared_error(y_train, train_pred))
    print("test mse:", mean_squared_error(y_test, test_pred))
    return model

#todo 1、sklearn
def xgboost_train(x_train,x_test,y_train,y_test):
    model = xgb.XGBRegressor(max_depth=3,
                     learning_rate=0.05,
                     n_estimators=100,
                     silent=True,
                     reg_alpha=1,
                     reg_lambda=1,
                     objective="reg:linear")
    model.fit(x_train,y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    print("train mse:",mean_squared_error(y_train, train_pred))
    print("test mse:",mean_squared_error(y_test, test_pred))

    return model

def combine_model(x_train,x_test,y_train,y_test,C):
    model1 = xgboost_train(x_train,x_test,y_train,y_test)
    model2 = lgb_tarin(x_train,x_test,y_train,y_test)
    from sklearn.linear_model import Ridge
    y_pred1 = model1.predict(x_train).ravel()
    y_pred2 = model2.predict(x_train).ravel()

    secend_x_train = pd.DataFrame({'y_pred1':y_pred1,'y_pred2':y_pred2})

    linear = Ridge(alpha=C)
    linear.fit(secend_x_train,y_train)
    secend_y_traiin = linear.predict(secend_x_train)
    print("train mse:",mean_squared_error(y_train, secend_y_traiin))

    y_predt1 = model1.predict(x_test).ravel()
    y_predt2 = model2.predict(x_test).ravel()
    secend_x_test = pd.DataFrame({'y_pred1':y_predt1,'y_pred2':y_predt2})
    secend_y_test = linear.predict(secend_x_test)
    print("test mse:",mean_squared_error(y_test, secend_y_test))


def main_sklearn():
    x,y = load_pandas(outline_path)
    clean(x,y)
    x_train, x_test, y_train, y_test = preprocession(x,y)
    #xgboost_train(x_train, x_test, y_train, y_test)
    #lgb_tarin(x_train, x_test, y_train, y_test)
    combine_model(x_train,x_test,y_train,y_test, 5)

if __name__=='__main__':
    main_sklearn()
