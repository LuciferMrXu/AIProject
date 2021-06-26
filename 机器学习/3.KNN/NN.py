#_*_ coding:utf-8_*_
import numpy as np

class NearestNeighbor():
    def __init__(self):
        pass

    # 记录所有训练数据
    def train(self,X,Y):
        self.Xtr=X
        self.Ytr=Y

    def predict(self,X):
        num_test=X.shape[0]
        Ypred=np.zeros(num_test,dtype=self.Ytr.dtype)

        # 对于每一个测试数据找出与其L1距离(曼哈顿距离)最小的样本的标签，作为其预测标签
        for i in range(num_test):
            distances=np.sum(np.abs(self.Xtr-X[i,:]),axis=1)
            min_index=np.argmin(distances)
            Ypred[i]=self.Ytr[min_index]

        return Ypred
