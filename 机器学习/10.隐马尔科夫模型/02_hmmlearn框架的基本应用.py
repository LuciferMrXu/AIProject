# -- encoding:utf-8 --
"""
Create by ibf on 2018/10/13
"""

import numpy as np
import hmmlearn.hmm as hmm

# 定义变量
states = ['盒子1', '盒子2', '盒子3']
obs = ['白球', '黑球']
n = 3
m = 2

# 训练数据
# 第一个序列：0,1,0,0,1
# 第二个序列：0,1,1,0,1
# 第三个序列：0,1,1
# 第四个序列：0,1,0,0,0
train = np.array([
    [0], [1], [0], [0], [1],
    [0], [1], [1], [0], [1],
    [0], [1], [1],
    [0], [1], [0], [0], [0]
])

# 构建模型
model = hmm.MultinomialHMM(n_components=n, n_iter=10, tol=0.01, random_state=28)

# 模型训练
model.fit(train, lengths=(5, 5, 3, 5))

# 查看一下模型训练的结果数据
print("模型训练得到的参数π:")
print(model.startprob_)
print("模型训练得到的参数A:")
print(model.transmat_)
print("模型训练得到的参数B:")
print(model.emissionprob_)

test = np.array([
    [0, 1, 0, 0, 1]  # 白，黑，白，白，黑
]).T
print("需要预测的序列:\n{}".format(test))
print("预测值为:{}".format(model.predict(test)))