# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/18
"""

import numpy as np


def entropy(t):
    """
    计算信息熵
    :param t:
    :return:
    """
    return np.sum([-p * np.log2(p) for p in t if p != 0])


def h1(y=[1, 1, 1, -1, -1, -1, 1, 1, 1, -1], w=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    """
    计算这个y中的信息熵
    :param y:
    :param w: 样本概率
    :return:
    """
    # 1. 计算y中取值为1的概率
    p1 = np.sum(np.asarray(w)[np.asarray(y) == 1])
    # 2. 计算y中取值为-1的概率
    p2 = 1 - p1
    # 3. 计算信息熵
    return entropy([p1, p2])


def h2(split=3, y=[1, 1, 1, -1, -1, -1, 1, 1, 1, -1], w=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    """
    根据给定的分割点对y进行数据划分
    :param split: 分割点，给定的是一个下标
    :param y:
    :param w:
    :return:
    """
    y = np.asarray(y)
    w = np.asarray(w)

    # 1. 计算左侧的相关信息
    data_y = y[:split]
    data_w = w[:split]
    p10 = np.sum(data_w)
    p11 = np.sum(data_w[data_y == 1]) / p10
    p12 = 1 - p11
    h1 = entropy([p11, p12])

    # 2. 计算右侧的相关信息
    data_y = y[split:]
    data_w = w[split:]
    p20 = np.sum(data_w)
    p21 = np.sum(data_w[data_y == 1]) / p20
    p22 = 1 - p21
    h2 = entropy([p21, p22])

    # 3. 计算划分的条件熵
    return p10 * h1 + p20 * h2


def h3(errs=[], w=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    """
    更新参数
    :param errs:给定异常的样本下标
    :param w:
    :return:
    """
    w = np.asarray(w)
    # 1. 计算错误率
    e = np.sum([w[i] for i in errs])
    # 2. 计算alpha
    alpha = 0.5 * np.log2((1 - e) / e)
    # 3. 计算更新后的权重
    # 获取预测成功的样本下标
    trues = [i for i in range(len(w)) if i not in errs]
    # 对预测成功的样本权重进行更新
    w[trues] = w[trues] * (np.e ** (-alpha))
    # 对预测失败的样本权重进行更新
    w[errs] = w[errs] * (np.e ** alpha)
    # 做一个归一化，保证np.sum(w) == 1.0
    w = w / np.sum(w)

    return e, alpha, w


def calc1():
    print("第一个子模型的选择")
    print("=" * 100)
    a = h1()
    print("原始数据的信息熵:{}".format(a))
    print("以2.5划分的信息增益:{}".format(a - h2(split=3)))
    print("以5.5划分的信息增益:{}".format(a - h2(split=6)))
    print("以8.5划分的信息增益:{}".format(a - h2(split=9)))
    err, alpha1, w = h3(errs=[6, 7, 8])
    print("第1个子模型的误差率:{}".format(err))
    print("第1个子模型的权重系数:{}".format(alpha1))
    print("第1个子模型更新的样本权重系数:\n{}\n".format(w))

    print("第二个子模型的选择")
    print("=" * 100)
    a = h1(w=w)
    print("原始数据的信息熵:{}".format(a))
    print("以2.5划分的信息增益:{}".format(a - h2(split=3, w=w)))
    print("以5.5划分的信息增益:{}".format(a - h2(split=6, w=w)))
    print("以8.5划分的信息增益:{}".format(a - h2(split=9, w=w)))
    err, alpha2, w = h3(errs=[0, 1, 2, 9], w=w)
    print("第2个子模型的误差率:{}".format(err))
    print("第2个子模型的权重系数:{}".format(alpha2))
    print("第2个子模型更新的样本权重系数:\n{}\n".format(w))

    print("第三个子模型的选择")
    print("=" * 100)
    a = h1(w=w)
    print("原始数据的信息熵:{}".format(a))
    print("以2.5划分的信息增益:{}".format(a - h2(split=3, w=w)))
    print("以5.5划分的信息增益:{}".format(a - h2(split=6, w=w)))
    print("以8.5划分的信息增益:{}".format(a - h2(split=9, w=w)))
    err, alpha3, w = h3(errs=[3, 4, 5], w=w)
    print("第3个子模型的误差率:{}".format(err))
    print("第3个子模型的权重系数:{}".format(alpha3))
    print("第3个子模型更新的样本权重系数:\n{}\n".format(w))

    print("第四个子模型的选择")
    print("=" * 100)
    a = h1(w=w)
    print("原始数据的信息熵:{}".format(a))
    print("以2.5划分的信息增益:{}".format(a - h2(split=3, w=w)))
    print("以5.5划分的信息增益:{}".format(a - h2(split=6, w=w)))
    print("以8.5划分的信息增益:{}".format(a - h2(split=9, w=w)))
    err, alpha4, w = h3(errs=[6, 7, 8], w=w)
    print("第3个子模型的误差率:{}".format(err))
    print("第3个子模型的权重系数:{}".format(alpha4))
    print("第3个子模型更新的样本权重系数:\n{}\n".format(w))

    print("=" * 100)
    print(alpha1)
    print(alpha2)
    print(alpha3)
    print(alpha4)

    print("=" * 100)
    print("1:{}".format(alpha1 - alpha2 + alpha3 + alpha4))
    print("2:{}".format(-alpha1 - alpha2 + alpha3 - alpha4))
    print("3:{}".format(-alpha1 + alpha2 + alpha3 - alpha4))
    print("4:{}".format(-alpha1 + alpha2 - alpha3 - alpha4))


calc1()
