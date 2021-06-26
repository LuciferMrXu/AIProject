# -- encoding:utf-8 --
"""
Create by ibf on 2018/10/18
"""

import numpy as np
import math


def log_sum_exp(a):
    """
    可以参考numpy中的log sum exp的API
    scipy.misc.logsumexp
    :param a:
    :return:
    """
    a = np.asarray(a)
    # a. 获取列表a中的最大值
    a_max = max(a)
    # b. 计算列表a中所有值和最大值之间的差值，做一个指数转换后的和
    tmp = 0
    for k in a:
        tmp += math.exp(k - a_max)
    # c. 将两部分合并得到最终结果
    return a_max + math.log(tmp)


def convert_obs_seq_2_index(Q, index=None):
    """
    根据传入的黑白文字序列转换为对应的索引值，如果是黑转换为1.如果是白转换为0.
    :param Q:
    :param index:
    :return:
    """
    if index is not None:
        cht = Q[index]
        if cht == '黑':
            return 1
        else:
            return 0
    else:
        result = []
        for q in Q:
            if q == '黑':
                result.append(1)
            else:
                result.append(0)
        return result
