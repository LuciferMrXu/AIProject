#_*_ coding:utf-8_*_
import numpy as np

def nonlin(x,deriv=False):
    if deriv==True:
        return x*(1-x)   # 反向传播公式
    #   选择sigmod函数作为激活函数
    return 1/(1+np.exp(-x))  # 前向传播公式


X=np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]
            ])
print(X.shape)

Y=np.array([[0],
            [1],
            [1],
            [0]])
print(Y.shape)

np.random.seed(16)

# 初始化权重参数
w0=2*np.random.random((3,4))-1
w1=2*np.random.random((4,1))-1

print(w0,w0.shape)
print(w1,w1.shape)

for i in range(60000):
    # 正向传播
    L0=X
    L1=nonlin(np.dot(L0,w0))
    L2=nonlin(np.dot(L1,w1))
    # 计算损失
    L2_error=L2-Y
    if i%10000==0:
        print('损失值：',np.mean(np.abs(L2_error)))
    # 反向传播
    L2_delta=L2_error*nonlin(L2,deriv=True)
    L1_error=L2_delta.dot(w1.T)
    L1_delta=L1_error*nonlin(L1,deriv=True)

    # 更新权重参数
    w1 -= L1.T.dot(L2_delta)
    w0 -= L0.T.dot(L1_delta)
