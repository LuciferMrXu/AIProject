import xgboost as xgb
import  pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
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
#读取完成之后直接可以做训练
def load_libsvm(path):
    dtrain = xgb.DMatrix(path)
    print(dtrain.get_label())

# todo 另外 libsvm sklearn也可以读取
def load_libsvm_sklearn(path):
    from sklearn.datasets import load_svmlight_file
    print(load_svmlight_file(path))

def clean(x,y):
    return 0
#todo 特征工程
def preprocession(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,test_size=0.2)
    x_train = x_train.drop(['V13','V26','V35','V36','V34'],axis=1)
    x_test = x_test.drop(['V13', 'V26', 'V35','V36','V34'], axis=1)
    poly=PolynomialFeatures(degree=2,include_bias=True,interaction_only=False)
    x_train=poly.fit_transform(x_train)
    x_test=poly.transform(x_test)
    return x_train,x_test,y_train,y_test





#todo 1、sklearn
def xgboost_train(x_train,x_test,y_train,y_test):
    model = xgb.XGBRegressor(max_depth=3,
                     learning_rate=0.1,
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
    '''
        输出特征重要性，删除重要性低的（去噪）
    '''
    # fm=model.feature_importances_
    # for ids,i in enumerate(fm):
    #     print('特征是%s，得分为%s'%(ids,i))
    return model




#todo 1、xgboost原生的训练方式
def xgboost_self(x_train,x_test,y_train,y_test):
    DTrain = xgb.DMatrix(data=x_train,label=y_train)
    DTest = xgb.DMatrix(data=x_test,label=y_test)
    param = {
        "objective":"reg:linear",
        "max_depth":3,
        "eta":0.06,
        "alpha":1,
        "lambda":1,
        "eval_metric":"rmse",
        'random_state' :16
    }
    xgb.train(params=param,dtrain=DTrain,num_boost_round=500,evals=[(DTrain,"train"),(DTest,"test")])

def main_sklearn():
    x,y = load_pandas(outline_path)
    clean(x,y)
    x_train, x_test, y_train, y_test = preprocession(x,y)
    xgboost_train(x_train, x_test, y_train, y_test)
    # xgboost_self(x_train, x_test, y_train, y_test)

if __name__=='__main__':
    #load_libsvm("./libsvm.txt")
    main_sklearn()
