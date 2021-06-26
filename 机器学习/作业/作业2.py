#_*_ coding:utf-8_*_
import numpy as np
import math
from sklearn.linear_model import RidgeCV, LassoCV,LinearRegression,Lasso,ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler


class Regress():
    def __init__(self,data,degree):
        self.data=data
        self.degree=degree

    def _preprocession(self):
        X,Y = np.hsplit(self.data,(13,))
        Y=Y.ravel()
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=16)
        # 归一化
        ss = MinMaxScaler()
        x_train = ss.fit_transform(x_train, y_train)
        x_test = ss.transform(x_test)
        # 特征维度拓展
        poly = PolynomialFeatures(degree=self.degree, include_bias=True, interaction_only=False)
        x_train = poly.fit_transform(x_train, y_train)
        x_test = poly.transform(x_test)
        linear = Lasso(alpha=0.5, fit_intercept=True, normalize=False, random_state=16)
        linear.fit(x_train, y_train)
        # 开始进行特征选择的操作
        final_column_indexs = []
        poly_column_num = x_train.shape[1]
        for index, coef in zip(range(poly_column_num), linear.coef_):
            if math.fabs(coef) > 1e-6:
                final_column_indexs.append(index)
            # else:
            #     print("删除第{}列的数据".format(index))
        x_train = x_train[:, final_column_indexs]
        x_test = x_test[:, final_column_indexs]

        return x_train, x_test, y_train, y_test


    def Linear(self):
        x_train, x_test, y_train, y_test=Regress._preprocession(self)
        model = LinearRegression(fit_intercept=True)
        model.fit(x_train, y_train)
        print("=" * 50)
        print("线性回归算法模型训练集上效果:{}".format(model.score(x_train, y_train)))
        print("线性回归算法模型测试集上效果:{}".format(model.score(x_test, y_test)))


    def Lasso(self,params=[1]):
        x_train, x_test, y_train, y_test=Regress._preprocession(self)
        model = LassoCV(alphas=params, cv=3,fit_intercept=True)
        model.fit(x_train, y_train)
        print("=" * 50)
        print("Lasso算法中的最优L1正则系数为：%s"%model.alpha_)
        print("Lasso算法模型训练集上效果:{}".format(model.score(x_train, y_train)))
        print("Lasso算法模型测试集上效果:{}".format(model.score(x_test, y_test)))


    def Ridge(self,params=[1]):
        x_train, x_test, y_train, y_test=Regress._preprocession(self)
        model = RidgeCV(alphas=params, cv=3,fit_intercept=True)
        model.fit(x_train, y_train)
        print("=" * 50)
        print("Ridge算法中的最优L2正则系数为：%s"%model.alpha_)
        print("Ridge算法模型训练集上效果:{}".format(model.score(x_train, y_train)))
        print("Ridge算法模型测试集上效果:{}".format(model.score(x_test, y_test)))


    def ElasticNet(self,params1=[1],params2=[1]):
        x_train, x_test, y_train, y_test = Regress._preprocession(self)
        model=ElasticNetCV(alphas=params2,l1_ratio=params1, cv=3,fit_intercept=True)
        model.fit(x_train, y_train)
        print("=" * 50)
        print("ElasticNet算法中的最优L1正则系数：%s，L2正则系数：%s"%(model.l1_ratio_,model.alpha_))
        print("ElasticNet算法模型训练集上效果:{}".format(model.score(x_train, y_train)))
        print("ElasticNet算法模型测试集上效果:{}".format(model.score(x_test, y_test)))


if __name__=='__main__':
    file = np.genfromtxt('boston_housing.data',dtype=float)
    Boston=Regress(file,7)
    Boston.Linear()
    Boston.Ridge([0.03, 0.01, 0.0075])
    Boston.Lasso([0.1, 0.08, 0.16])
    Boston.ElasticNet([0.1, 0.5, 1.0],[0.1, 0.5, 1.0])
