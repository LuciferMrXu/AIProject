# --encoding:utf-8 --
"""
使用SMOTE算法进行数据合成：
SMOTE算法的原理就是在生成新的小众样本数据的时候，新的样本点的坐标是两个旧样本点（选择的是比较相似的两个旧样本点，换句话来讲就是选距离比较近的旧样本点）连线上的某一个坐标值
SMOTE算法和上采样的方式比较接近，都是增加小众样本数据的数据量；只是上采样中，是直接抽样增加；而在SMOTE算法中，是通过某种数据合成规则来构建新的样本点
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 给定随机种子
np.random.seed(28)


class Smote:
    def __init__(self, samples, N=10, k=5):
        """
        samples是DataFrame对象，是原始的小众数据样本
        N: 每个样本需要扩展的几个其它样本
        k: 计算近邻的时候，邻居的数量
        """
        self.n_samples, self.n_attrs = samples.shape
        self.N = N
        self.k = k
        self.samples = samples

    def over_sampling(self):
        # 每个类别至少合成的样本数量
        self.synthetic = pd.DataFrame(columns=self.samples.columns)
        self.new_index = 0

        # 模型训练
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)

        # 对所有样本进行一个遍历，每个样本都需要产生N个随机的新样本
        for i in range(len(self.samples)):
            # 针对于当前样本，获取对应的k个邻居的索引下标
            nnarray = neighbors.kneighbors([self.samples.iloc[i]], return_distance=False)
            # 存储具体的数据
            self.__populate(self.N, i, nnarray.flatten())

        return self.synthetic

    # 从k个邻居中随机选取N次，生产N个合成的样本
    def __populate(self, N, i, nnarray):
        for j in range(N):
            # 获取随机一个索引值（为了获取相似样本的坐标值）
            nn = np.random.randint(0, self.k)
            # 计算两个样本之间的距离/其实是坐标的差值
            dif = self.samples.iloc[nnarray[nn]] - self.samples.iloc[i]
            # 随机一个每个维度上的随机数量
            gap = np.random.rand(1, self.n_attrs)
            # 进行累加操作
            self.synthetic.loc[self.new_index] = self.samples.iloc[i] + gap.flatten() * dif
            self.new_index += 1


if __name__ == '__main__':
    # 1. 模拟数据创建
    category1 = np.random.randint(0, 10, [10000, 5]).astype(np.float)
    label1 = np.array([1] * 10000).reshape(-1, 1)
    data1 = np.concatenate((category1, label1), axis=1)
    category2 = np.random.randint(8, 18, [10, 5]).astype(np.float)
    label2 = np.array([0] * 10).reshape(-1, 1)
    data2 = np.concatenate((category2, label2), axis=1)

    name = ['A', 'B', 'C', 'D', 'E', 'Label']
    data = np.concatenate((data1, data2), axis=0)
    df = pd.DataFrame(data, columns=name)
    print(df.head())

    # 2. 查看各个类别的数据
    print("=" * 100)
    print(df.Label.value_counts())

    # 3. 获取小众类别的数据
    small_category = df[df.Label == 0.0]
    print("=" * 100)
    print(small_category.head())

    # 4. 进行数据合成增加
    N = 20
    k = 3
    smote = Smote(small_category, N, k)
    over_sample_data = smote.over_sampling()
    print("=" * 100)
    print(over_sample_data)

    # 5. 合并数据
    final_df = pd.concat([df, over_sample_data], ignore_index=True)
    print("=" * 100)
    print(final_df.head())
    print("=" * 100)
    print(final_df.Label.value_counts())
    print("=" * 100)
    print(final_df.describe())
    print("数据样本增加后的小众样本数据的特征描述信息:")
    print(final_df[final_df.Label == 0.0].describe())
    print("数据样本增加前的小众样本数据的特征描述信息:")
    print(df[df.Label == 0.0].describe())
    print("大众样本数据的特征描述信息:")
    print(final_df[final_df.Label == 1.0].describe())

    # TODO: 大家可以自己讲X的维度设置为2，然后进行画图展示
