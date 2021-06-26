#_*_ coding:utf-8_*_
import numpy as np
from sklearn.linear_model import LinearRegression


# 构建模型数据
'''
        数据一
'''
x = np.array([
        [1,2],
        [2,3],
        [3,4],
        [4,5],
        [5,6],
        [7,8],
        [9,10]
])
y = np.array([
        [11],
        [12],
        [13],
        [14],
        [15],
        [16],
        [17]
])
'''
        数据二
'''
# np.random.seed(16)
# N=10
# x=np.linspace(0,6,N)+np.random.randn(N)
# y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.randn(N)
# x = x.reshape(-1, 1)
# y = np.reshape(y, (-1, 1))
# print(x.shape)
# print(y.shape)

model = LinearRegression()
model.fit(x, y.ravel())
s1=model.score(x,y)
print('模块自带线性回归默认实现模型=============')
print("模型自带score API评估值为:{}".format(s1))
print("参数列表为:{}".format(model.coef_))
print("截距项为:{}".format(model.intercept_))




def lingear(X,Y,alpha=0.002,max_iter=-1,tol=1e-10, fit_intercept=True):
        # 1. 校验一下X和Y的格式是否正常
        assert validate(X, Y)

        X = np.array(X)
        Y = np.array(Y)
        # 2. 开始获取相关参数
        # 获取样本数和维度数目
        m, n = np.shape(X)
        # 定义theta参数
        theta = np.zeros((n,1))
        # 定义截距项
        intercept = 0
        # 获取最大允许迭代次数
        max_iter = 100000 if max_iter <= 0 else max_iter
        # 构建一个误差保存的对象
        loss_old=[]

        for round in range(max_iter):
                if fit_intercept:
                        theta = theta + x.T.dot(y - x.dot(theta)+intercept) * alpha
                        loss = np.power(y - x.dot(theta), 2).sum()

                        gd = np.sum(loss_old)
                        # 进行参数模型更新
                        intercept += alpha * gd
                else:
                        theta = theta + x.T.dot(y - x.dot(theta)) * alpha
                        loss = np.power(y - x.dot(theta), 2).sum()
                        loss_old.append(loss)

                # 停止条件
                if loss - loss_old[-1] <= tol and loss - loss_old[-1] >= -tol:
                        break


        return theta, intercept


def validate(X, Y):
    """
    校验X和Y的格式是否正常，如果不正常，返回False；否则返回True
    :param X:
    :param Y:
    :return:
    """
    m1, n1 = np.shape(X)
    m2, n2 = np.shape(Y)
    if m1 != m2:
        return False
    else:
        if n2 != 1:
            return False
        else:
            return True





if __name__=='__main__':
        theta, intercept= lingear(x, y)
        print("参数为：{}".format(theta))
        print("截距项为:{}".format(intercept))












