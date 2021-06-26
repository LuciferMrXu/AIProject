#_*_ coding:utf-8_*_
import numpy as np
from sklearn import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


class LinearRegression():
    def __init__(self):
        self.w=None

    def fit(self,X,y):
        print(X.shape)
        X=np.insert(X,0,1,axis=1)       # 为偏置权重(截距项)插入常数1
        print(X.shape)
        X_=np.linalg.inv(X.T.dot(X))    # 求逆矩阵(X_可以从外部访问)
        self.w=X_.dot(X.T).dot(y)

    def predict(self,X):
        X = np.insert(X, 0, 1, axis=1)  # 为偏置权重(截距项)插入常数1
        y_pred=X.dot(self.w)
        return y_pred

def mean_squared_error(y_true,y_pred):
    mse=np.mean(np.power(y_true-y_pred,2))
    return mse

def main():
    # 加载数据
    data1=datasets.load_diabetes()   # 本地加载糖尿病数据集

    X=data1.data[:,np.newaxis,2]     # 只使用一组特征
    print(X.shape)

    # 将数据划分为训练集和测试集
    x_train,x_test=X[:-20],X[-20:]
    # 将目标划分为训练集和测试集
    y_train,y_test=data1.target[:-20],data1.target[-20:]

    clf=LinearRegression()
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)

    print('MSE:',mean_squared_error(y_test,y_pred))

    # 画图
    plt.scatter(x_test[:,0],y_test,color='r')
    plt.plot(x_test[:,0],y_pred,color='b',linewidth=3)
    plt.show()


if __name__=='__main__':
    main()


