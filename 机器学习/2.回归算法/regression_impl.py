import numpy as np
from icecream import ic




class Regression:
    def __init__(self,kind,lr,batch):
        self.kind = kind
        self.lr = lr
        self.batch = batch


    def __call__(self,X, weights, bias, y):
        self.mian(X, weights, bias, y)

    # 损失函数MSE
    def mse_loss(self,yhats, y):
        return np.mean( (yhats - y) ** 2 )

    # loss对于w的偏导
    def partial_w(self,yhats, y, train_x):
        # axis=0表示按行计算，得到列的性质
        return 2 * np.mean((yhats - y) * train_x, axis=0)

    # loss对于b的偏导
    def partial_b(self,yhats, y):
        return 2 * np.mean(yhats - y)

    # sigmoid函数，二分类变换
    def logistic(self,x):
        return 1 / (1 + np.exp(-x))


    # def softmax(self,x):
    #     # 归一化，缩放x
    #     x -= np.max(x)
    #     sum = np.sum(np.exp(x))
    #     return np.exp(x) / sum


    # softmax函数，多分类变换
    def softmax(self,x):
        x -= np.max(x, axis=1).reshape(x.shape[0], 1)
        x_sum = np.sum(np.exp(x), axis=1)
        return np.exp(x) / x_sum.reshape(x.shape[0], 1)

    # 多分类的loss函数
    def cross_entropy(self,yhats, y):
        return - np.mean( y * np.log(yhats))

    # 线性回归
    def train_linear_regression(self,X, weights, bias, y):
        for i in range(self.batch):
            yhats = X @ weights.T + bias
            loss_value = self.mse_loss(yhats, y)
            ic(i,loss_value)
            learning_rate = self.lr
            weights += -1 * self.partial_w(yhats, y, X) * learning_rate
            bias += -1 * self.partial_b(yhats, y)


    # 逻辑回归
    def train_logistic_regression(self,X, weights, bias, y):
        for i in range(self.batch):
            # 对输出的y做sigmoid变换，转换为0-1的概率
            yhats = self.logistic(X @ weights.T + bias)
            loss_value = self.mse_loss(yhats, y)
            # 定义阈值
            threshold = 0.5
            probs = np.array((yhats > threshold), dtype=np.int)
            ic(i,loss_value,probs,yhats)
            learning_rate = self.lr
            weights += -1 * self.partial_w(yhats, y, X) * learning_rate
            bias += -1 * self.partial_b(yhats, y)


    # 多分类回归
    def train_cross_entropy(self,X, weights, bias, y):
        for i in range(self.batch):
            # 先对label做softmax变换
            yhats = self.softmax(X @ weights.T + bias)
            loss_value = self.cross_entropy(yhats, y)
            ic(i,loss_value,yhats)
            learning_rate = self.lr
            # weights += -1 * self.partial_w(yhats, y, X) * learning_rate
            # bias += -1 * self.partial_b(yhats, y)


    def mian(self,X, weights, bias, y):
        if self.kind == 1:
            self.train_linear_regression(X, weights, bias, y)
        elif self.kind == 2:
            self.train_logistic_regression(X, weights, bias, y)
        elif self.kind ==3:
            self.train_cross_entropy(X, weights, bias, y)

if __name__ == '__main__':
    X = np.random.normal(size=(10, 7))
    y_fenlei = np.array([
        [1],
        [0],
        [0],
        [0],
        [1],
        [0],
        [0],
        [1],
        [0],
        [0],
    ])
    y_duofenlei = np.array([
        [1,0,0,0,0],
        [0,0,0,1,0],
        [0,1,0,0,0],
        [0,0,0,1,0],
        [1,0,0,0,0],
        [0,0,0,0,1],
        [0,1,0,0,0],
        [0,0,0,1,0],
        [0,0,1,0,0],
        [0,0,0,0,1],
    ])
    y_huigui = np.random.random(size=(10,1))
    weights_duo = np.random.normal(size=(5,7))

    weights = np.random.normal(size=(1,7))
    bias = 0
    # @表示两个矩阵相乘，通过运算符重载实现
    # ic(X @ weights + bias)
    res = Regression(3,1e-3,100)
    # res(X, weights, bias, y_huigui)
    # res(X, weights, bias, y_fenlei)
    res(X, weights_duo, bias, y_duofenlei)
