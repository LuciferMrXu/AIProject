#_*_ coding:utf-8_*_
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler,PolynomialFeatures


def load_data(path):
    fr=pd.read_csv(path,sep=';',low_memory=False)
    # print(fr.head())
    # print(fr.info())
    # print(fr.describe())
    fr = fr.replace('?', np.nan)
    fr = fr.dropna(axis=0, how='any')
    return fr


def preprocession(df):
    # 过滤异常样本点
    sample = df.iloc[:, 2:4].astype(float)
    IF=IsolationForest(n_estimators=75,contamination="auto",behaviour='new')
    IF.fit(sample)
    pre=IF.predict(sample)
    df=df[pre==1]
    # 划分训练集测试集
    x_train = df.iloc[:, 2:4].astype(float)
    y_train = df.iloc[:, 5].astype(float)
    df_text=pd.DataFrame.sample(df,n=165,random_state=7)
    x_test=df_text.iloc[:, 2:4].astype(float)
    y_test = df_text.iloc[:, 5].astype(float)
    # 标准化
    ss=StandardScaler()
    x_train=ss.fit_transform(x_train)
    x_test=ss.transform(x_test)
    # 特征维度拓展
    poly=PolynomialFeatures(degree=3,include_bias=True,interaction_only=False)
    x_train=poly.fit_transform(x_train)
    x_test=poly.transform(x_test)

    return x_train,x_test,y_train,y_test


def xgboost(x_train, x_test, y_train, y_test):
    algo = xgb.XGBRegressor(gpu_id=0,max_bin=16,tree_method='gpu_hist',
                     max_depth=3,
                     learning_rate=0.5,
                     n_estimators=500,
                     silent=True,
                     reg_alpha=1,
                     reg_lambda=1,
                     objective="reg:linear")
    algo.fit(x_train,y_train)
    test_pred = algo.predict(x_test)
    print("train score:",algo.score(x_train,y_train))
    print("test score:",algo.score(x_test, y_test))

    return test_pred


def draw(x_test,y_test,test_pred):
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    t = np.arange(len(x_test))
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实值')
    plt.plot(t, test_pred, 'g-', linewidth=2, label=u'预测值')
    plt.legend(loc='lower right')
    plt.title(u"线性回归预测功率与电流之间的关系", fontsize=20)
    plt.grid(b=True)
    plt.show()


def main():
    df=load_data(path)
    x_train, x_test, y_train, y_test=preprocession(df)
    test_pred=xgboost(x_train, x_test, y_train, y_test)
    draw(x_test, y_test,test_pred)


if __name__=='__main__':
    path='household_power_consumption.txt'
    main()
