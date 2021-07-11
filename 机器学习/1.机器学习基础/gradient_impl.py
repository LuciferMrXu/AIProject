'''
    模拟梯度下降
'''
import random
from icecream import ic


class Gradient:
    def __init__(self,lr):
        # 学习率，防止梯度爆炸
        self.lr = lr


    def __call__(self,w, x, b, y):
        self.main(self.lr,w, x, b, y)

    # 平方和损失函数
    def loss(self,xs, w, b, ys):
        return ((xs * w + b) - ys) ** 2

    # 对loss求导后的结果
    def gradient(self,w, x, b, y):
        return 2 * (w * x + b - y) * x


    def main(self,lr,w, x, b, y):
        for _ in range(100):
            w_gradient = self.gradient(w, x, b, y)

            w = w + -1 * w_gradient * lr

            ic(w)
            ic(self.loss(x, w, b, y))

if __name__ == '__main__':
    lr = 1e-3

    w, b = random.randint(-10, 10), random.randint(-10, 10)

    x, y = 10, 0.35

    result = Gradient(lr)
    result(w, x, b, y)

    