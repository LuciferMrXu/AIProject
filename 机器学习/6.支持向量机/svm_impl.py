import numpy as np
import matplotlib.pyplot as plt


'''
    感知机模型
'''
class Perceptron:
    def __init__(self) -> None:
        pass

    def __call__(self, k_and_b,label_a_x,label_b_x):
        self.main(k_and_b,label_a_x,label_b_x)

    # 一元一次方程：y=kx+b
    def f(self,x, k, b):
        return k * x + b


    def main(self,k_and_b,label_a_x,label_b_x):
        # 生成100个随机的k和b
        for i in range(100):
            k, b = (np.random.random(size=(1, 2)) * 10 - 5)[0]
            print(k, b)
            # 找到能满足把两个类别数据分割出来的k和b
            if np.max(self.f(label_a_x, k, b)) <= -1 and np.min(self.f(label_b_x, k, b)) >= 1:
                print(k, b)
                k_and_b.append((k, b))

        x = np.concatenate((label_a_x, label_b_x))

        for k, b in k_and_b:
            plt.plot(x, self.f(x, k, b))
        # 找到分割效果最好的k和b（找距离两个类别所有点距离最远的直线）
        k_star, b_star = sorted(k_and_b, key=lambda t: abs(t[0]))[0]

        plt.plot(x, self.f(x, k_star, b_star), '-o')

        

if __name__ == '__main__':
    label_a = np.random.normal(6, 2, size=(50, 2))
    label_b = np.random.normal(-6, 2, size=(50, 2))
    label_a_x = label_a[:, 0]
    label_b_x = label_b[:, 0]
    plt.scatter(*zip(*label_a))
    plt.scatter(*zip(*label_b))

    k_and_b = []

    simple_svm = Perceptron()
    simple_svm(k_and_b,label_a_x,label_b_x)

    plt.show()