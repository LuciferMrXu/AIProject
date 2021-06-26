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

# 自定义的模型参数
start_probability = np.array([0.2, 0.5, 0.3])
transition_probability = np.array([
    [0.5, 0.4, 0.1],
    [0.2, 0.2, 0.6],
    [0.2, 0.5, 0.3]
])
emission_probability = np.array([
    [0.4, 0.6],
    [0.8, 0.2],
    [0.5, 0.5]
])

# 定义模型
model = hmm.MultinomialHMM(n_components=n)

# 明确给定模型参数
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# 做预测，也就是viterbi
test = np.array([
    [0, 1, 0, 0, 1]  # 白，黑，白，白，黑
]).T
print("需要预测的序列:\n{}".format(test))
print("预测值为:{}".format(model.predict(test)))
print("各个预测值的概率（这个里面不是使用viterbi算法计算的，使用的是单状态概率伽玛来计算）:\n{}".format(model.predict_proba(test)))
logprod, box_index = model.decode(test, algorithm='viterbi')
print("预测的盒子序号:{}".format(box_index))
print("预测为该状态序列的概率为:{}".format(np.exp(logprod)))
