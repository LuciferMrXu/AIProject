import matplotlib.pyplot as plt
import numpy as np
import random
from icecream import ic
from collections import defaultdict
from matplotlib.colors import BASE_COLORS
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False ## 设置正常显示符号



def loadDataSet(fileName):      # 解析制表符分隔的浮点数的常规函数
    fr=np.genfromtxt(fileName, delimiter="\t", dtype=float)
    plt.figure(facecolor='w')
    plt.scatter(*zip(*fr), color='r', label='数据分布')
    plt.legend(loc='best')
    plt.title("K-Means", fontsize=20)
    plt.show()

    return fr


class Kmeans:
    def __init__(self,k,threshold,data):
        # 1、设定k的值
        self.k = k
        # 设定阈值
        self.threshold = threshold
        self.data = data

    def __call__(self):
        self.main(self.k,self.data)

    # 2、在数据中随机初始化k个点
    def random_centers(self,k, points):
        for i in range(k):
            yield random.choice(points[:, 0]), random.choice(points[:, 1])

    # 3、计算离各中心点所有最近点的均值
    def mean(self,points):
        all_x, all_y = [x for x, y in points], [y for x, y in points]

        return np.mean(all_x), np.mean(all_y)

    # 欧几里得距离计算公式
    def distance(self,p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        return np.sqrt((x1 - x2) ** 2 + (y1 - y2)**2)


    def main(self,k, points, centers=None):
        colors = list(BASE_COLORS.values())

        if not centers:
            centers = list(self.random_centers(k=k, points=points))

        ic(centers)

        for i, c in enumerate(centers):
            plt.scatter([c[0]], [c[1]], s=90, marker='*', c=[colors[i]])

        # python拆包语法
        plt.scatter(*zip(*points), c='black')

        centers_neighbor = defaultdict(set)

        # 分别找每一个点p分别离哪一个中心点c最近
        for p in points:
            closet_c = min(centers, key=lambda c: self.distance(p, c))
            centers_neighbor[closet_c].add(tuple(p))

        for i, c in enumerate(centers):
            _points = centers_neighbor[c]
            all_x, all_y = [x for x, y in _points], [y for x, y in _points]
            plt.scatter(all_x, all_y, c=[colors[i]])

        plt.show()

        new_centers = []
        # 计算每个类所有点的平均值，重新定位新中心点
        for c in centers_neighbor:
            new_c = self.mean(centers_neighbor[c])
            new_centers.append(new_c)

        self.threshold = 1
        distances_old_and_new = [self.distance(c_old, c_new) for c_old, c_new in zip(centers, new_centers)]
        ic(distances_old_and_new)
        # 递归调用，当新计算出的中心点c和上一轮的c相比小于自定义阈值threshold，停止迭代，最终找到各个聚类。
        if all(c < self.threshold for c in distances_old_and_new):
            return centers_neighbor
        else:
            self.main(k, points, new_centers)

if __name__ == '__main__':
    # points0 = np.random.normal(size=(100, 2))
    # points1 = np.random.normal(loc=1, size=(100, 2))
    # points2 = np.random.normal(loc=2, size=(100, 2))
    # points3 = np.random.normal(loc=5, size=(100, 2))

    # points = np.concatenate([points0, points1, points2, points3])


    # # python拆包语法
    # ic(*zip(*points0))

    dataMat = np.asarray(loadDataSet('./机器学习/7.聚类算法/datas/testSet.txt'))


    result = Kmeans(4, 0.1,data=dataMat)
    result()
