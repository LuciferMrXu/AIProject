# -- encoding:utf-8 --
"""
HashTF: 以单词的hash值作为特征属性
Create by ibf on 2018/10/14
"""

import numpy as np

data1 = open('./d1.txt', 'r').read()
data2 = open('./d2.txt', 'r').read()
data = []
item1 = []
for items in data1.split('\n'):
    for item in items.split(' '):
        item1.append(item)
item2 = []
for items in data2.split('\n'):
    for item in items.split(' '):
        item2.append(item)

data.append(item1)
data.append(item2)
print("原始数据:\n{}".format(np.array(data)))

# b. 定义最多允许存在的多少个特征属性
max_features = 10
print("最大允许的特征属性数目:{}".format(max_features))

# b 遍历所有文档统计出现的次数
features = []
for doc in data:
    # 1. 统计当前文档中，各个单词出现的次数，也就是各个特征属性出现次数
    result = {}
    for word in doc:
        feature_name = hash(word) % max_features
        if feature_name not in result:
            result[feature_name] = 1
        else:
            result[feature_name] += 1
    # 2. 产生最终的特征属性
    feature = []
    for feature_name in range(max_features):
        if feature_name not in result:
            feature.append(0)
        else:
            feature.append(result[feature_name])
    features.append(feature)
print("最终得到的特征属性矩阵:\n{}".format(np.array(features)))
print(np.shape(features))


