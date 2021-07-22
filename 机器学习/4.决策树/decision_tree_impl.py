from icecream import ic
import numpy as np
import pandas as pd
from collections import Counter



class DecisionTree:
    def __init__(self) -> None:
        pass

    def __call__(self,training_dataset, target ):
        self.find_best_split(training_dataset, target)

    # 获取概率
    def get_pros(self,elements):
        counter = Counter(elements)
        pr = np.array([counter[c] / len(elements) for c in counter])

        return pr

    # 信息熵，信息的混乱程度
    def entropy(self,elements):
        pr = self.get_pros(elements)
        return -np.sum(pr * np.log2(pr))

    # 基尼系数
    def gini(self,elements):
        pr = self.get_pros(elements)
        return 1 - np.sum(pr**2)



    # 分类决策树的loss
    def cart_loss(self,left, right, pure_fn):
        m_left, m_right = len(left), len(right)
        m = m_left + m_right

        return m_left / m * pure_fn(left) + m_right / m * pure_fn(right)


    # 找到所有特征中的最优特征和最好划分方式
    def find_best_split(self,training_dataset, target):
        dataset = training_dataset
        # 输入数据的特征种类
        fields = set(dataset.columns.tolist()) - {target}
        ic(fields)

        mini_loss = float('inf')
        best_feature, best_split = None, None

        for x in fields:
            filed_value = dataset[x]
            for v in filed_value:
                # 左子树的数据
                split_left = dataset[dataset[x] == v][target].tolist()
                # 右子树的数据
                split_right = dataset[dataset[x] != v][target].tolist()

                loss = self.cart_loss(split_left, split_right, pure_fn=self.gini)
                ic(x, v, self.cart_loss(split_left, split_right, pure_fn=self.gini))
                if loss < mini_loss:
                    best_feature, best_split = x, v

        return best_feature, best_split


if __name__ == '__main__':
    dt = DecisionTree()

    ic(dt.entropy([1, 1]))
    ic(dt.gini([1, 1]))
    ic(dt.entropy([0, 0]))
    ic(dt.gini([0, 0]))
    ic(dt.entropy([0, 0, 1, 1, 1,1 ,1, 1]))
    ic(dt.gini([0, 0, 1, 1, 1,1 ,1, 1]))
    ic(dt.entropy([0, 0, 0, 0, 0, 0, 0, 0]))
    ic(dt.gini([0, 0, 0, 0, 0, 0, 0, 0]))
    ic(dt.entropy([1, 2, 3, 4, 56, 7, 8, 1, 19]))
    ic(dt.gini([1, 2, 3, 4, 56, 7, 8, 1, 19]))
    ic(dt.entropy([1, 2, 3, 4, 65, 76, 87, 32, 21]))
    ic(dt.gini([1, 2, 3, 4, 65, 76, 87, 32, 21]))



    sales = {
        'gender': ['Female', 'Female', 'Female', 'Female', 'Male', 'Male', 'Male'],
        'income': ['H', 'M', 'H', 'M', 'H', 'H', 'L'],
        'family-number': [1, 1, 2, 1, 1, 1, 2],
        'bought': [1, 1, 1, 0, 0, 0, 1]
    }

    sales_dataset = pd.DataFrame.from_dict(sales)
    target = 'bought'
    ic(dt(sales_dataset, target='bought'))