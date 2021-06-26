# -- encoding:utf-8 --
"""
词袋法：一种将单词数据转换为特征向量的方式，以单词作为特征属性，以单词在当前文本中出现的次数作为特征值来构建的一个方式
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
print(data)

# 统计一下各个单词在当前文本中出现的数目
# a. 统计一下有哪些单词 --> 使用哪些单词作为特征属性
feature_names = set()
for doc in data:
    for word in doc:
        feature_names.add(word)
feature_names = list(feature_names)
print(feature_names)

# b 遍历所有文档统计出现的次数
features = []
for doc in data:
    # 1. 统计当前文档中，各个单词出现的次数，也就是各个特征属性出现次数
    result = {}
    for word in doc:
        if word not in result:
            result[word] = 1
        else:
            result[word] += 1
    # 2. 产生最终的特征属性
    feature = []
    for feature_name in feature_names:
        if feature_name not in result:
            feature.append(0)
        else:
            feature.append(result[feature_name])
    print(feature)
    features.append(feature)
print(features)
print(np.shape(features))


