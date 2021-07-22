from sklearn.datasets import load_iris
from icecream import ic
from collections import Counter
import numpy as np



def knn(x, y, query, k=3, clf=True):
    history = {tuple(x_): y_ for x_, y_ in zip(x, y)}
    # 计算所有已知数据中离预测数据最近的k个点
    neighbors = sorted(history.items(), key=lambda x_y: np.sum((np.array(x_y[0]) - np.array(query)) ** 2))[:k]
    neighbors_y = [y for x, y in neighbors]

    if clf: 
        #分类问题
        return Counter(neighbors_y).most_common(1)[0][0]
    else:
        #回归问题
        return np.mean(neighbors_y)


if __name__ == '__main__':
    iris_x = load_iris()['data']
    iris_y = load_iris()['target']
    ic(iris_y)
    query = [6.5,3.3,6,2.2]
    ic(knn(iris_x,iris_y,query,k=5))
