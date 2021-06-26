# -- encoding:utf-8 --
"""
Create by ibf on 2018/9/16
"""
# http://scipy.github.io/devdocs/generated/scipy.stats.multivariate_normal.html#scipy.stats.multivariate_normal

import numpy as np
from scipy.stats import multivariate_normal

# 1. 产生模拟数据
# TODO: 自己去把data的数据用图像展示一下，看看是不是服从高斯分布(数据是不是在一团)
np.random.seed(28)
N1 = 400
N2 = 100
# 类别1的数据
mean1 = (0, 0, 0)
cov1 = np.diag((1, 2, 3))
data1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=N1)
# 类别2的数据
mean2 = (5, 5, 5)
cov2 = np.array([
    [3, 1, 0],
    [1, 1, 0],
    [0, 0, 5]
])
data2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=N2)
# 合并两个数据
data = np.vstack((data1, data2))
# 构建对应的

# 2. 模型构建
num_iter = 1000
m, d = data.shape
print("样本数目:{}，每个样本的维度:{}".format(m, d))

# 给定初始化的均值、方差和pi值
mu1 = data.min(axis=0)
mu2 = data.max(axis=0)
sigma1 = np.identity(d)
sigma2 = np.identity(d)
pi = 0.5
print("初始化的均值:\n{}\n{}".format(mu1, mu2))
print("初始化的方差:\n{}\n{}".format(sigma1, sigma2))
print("初始化的概率值:\n{}\n{}".format(pi, 1 - pi))

# 实现EM算法
for i in range(num_iter):
    # 1. E step：计算在当前模型参数的情况下，各个样本的条件概率
    # a. 根据均值和方差构建对应的多元高斯概率密度函数
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    # b. 基于多元高斯概率密度函数计算样本的概率值
    pdf1 = pi * norm1.pdf(data)
    pdf2 = [1 - pi] * norm2.pdf(data)
    # c. 计算w，做一个归一化的操作
    w1 = pdf1 / (pdf1 + pdf2)
    w2 = 1 - w1

    # 2. M Step: 基于样本概率更新模型参数
    # a. 均值更新
    mu1 = np.dot(w1, data) / np.sum(w1)
    mu2 = np.dot(w2, data) / np.sum(w2)
    # b. 方差更新
    sigma1 = np.dot(w1 * (data - mu1).T, data - mu1) / np.sum(w1)
    sigma2 = np.dot(w2 * (data - mu2).T, data - mu2) / np.sum(w2)
    # c. π的更新
    pi = np.sum(w1) / m

print("最终输出的均值:\n{}\n{}".format(mu1, mu2))
print("最终输出的方差:\n{}\n{}".format(sigma1, sigma2))
print("最终输出的概率值:\n{}\n{}".format(pi, 1 - pi))

# 3. 基于训练好的模型参数对数据做一个预测(数据的聚类划分)
x_test = np.array([
    [0.0, 0.0, 0.0],
    [2.5, 1.5, 1.5],
    [5.0, 5.0, 5.0],
    [6.0, 5.0, 7.0]
])
# a. 基于构建好的均值和方差构建一个多元高斯分布的概率密度函数
norm1 = multivariate_normal(mu1, sigma1)
norm2 = multivariate_normal(mu2, sigma2)
# b. 基于多元高斯概率密度函数计算样本的概率值
pdf1 = pi * norm1.pdf(x_test)
pdf2 = [1 - pi] * norm2.pdf(x_test)
# c. 计算w，做一个归一化的操作
w1 = pdf1 / (pdf1 + pdf2)
w2 = 1 - w1
print("预测为类别1的概率为:{}".format(w1))
print("预测为类别2的概率为:{}".format(w2))
w = np.vstack((w1, w2))
y_hat = np.argmax(w, axis=0)
print("预测的类别为:{}".format(y_hat))
