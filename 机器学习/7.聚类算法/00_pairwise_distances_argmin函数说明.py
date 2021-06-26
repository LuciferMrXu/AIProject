# -- encoding:utf-8 --
"""
Create by ibf on 2018/9/8
"""

from sklearn.metrics.pairwise import pairwise_distances_argmin

"""
pairwise_distances_argmin: 主要用于计算两个集合中的数据距离最近的映射关系
pairwise_distances_argmin(A, B): 结果返回对于集合A中的每个样本，距离最近的B集合数据的索引下标
"""

print(pairwise_distances_argmin([[5], [3], [2]], [[1.5], [3.5], [6]]))
print(pairwise_distances_argmin([[5], [3], [2]], [[2.5], [3.6], [6]]))
print(pairwise_distances_argmin([[5, 1], [3, 2], [2, 3]], [[2.5, 1.5], [3.6, 2], [6, 1]]))
