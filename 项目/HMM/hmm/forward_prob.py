# -- encoding:utf-8 --
"""
前向算法
Create by ibf on 2018/10/18
"""

from wzx.hmm import common
import numpy as np


def calc_alpha(pi, A, B, Q, alpha, fetch_index_by_obs_seq=None):
    """
    计算前向概率α的值
    :param pi:  给定的模型的初始状态概率向量
    :param A:  给定的模型状态之间的转移概率矩阵
    :param B:  给定的模型状态与观测值之间的转移概率矩阵
    :param Q:  观测值序列组成的一个向量/数组/集合
    :param alpha:  是需要更新的一个前向概率矩阵
    :param fetch_index_by_obs_seq:  根据序列参数返回对应索引值的函数，可以为None
    :return:  更新完成后，返回alpha参数
    """
    # 1. 获取索引值的方法初始化
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        # 如果在调用方法的时候没有给定序列转换索引的方式，那么就使用默认的转换方式
        # 默认的时候使用字符的ASCII码
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 2. 定义状态相关信息
    n = np.shape(A)[0]
    T = np.shape(Q)[0]
    n_range = range(n)

    # 3. 更新t=0时刻对应的前向概率的值
    for i in n_range:
        alpha[0][i] = pi[i] * B[i][fetch_index_by_obs_seq_f(Q, 0)]

    # 4. 更新t=1到t=T-1时刻对应的前向概率的值
    for t in range(1, T):
        for i in n_range:
            # a. 获取上一个时刻到当前时刻的状态转移概率
            tmp_prob = 0.0
            for j in n_range:
                tmp_prob += alpha[t - 1][j] * A[j][i]

            # b. 更新当前时刻t的对应前向概率值
            alpha[t][i] = tmp_prob * B[i][fetch_index_by_obs_seq_f(Q, t)]

    # 5. 返回最终的更新值
    return alpha


if __name__ == '__main__':
    pi = np.array([0.2, 0.5, 0.3])
    A = np.array([
        [0.5, 0.4, 0.1],
        [0.2, 0.2, 0.6],
        [0.2, 0.5, 0.3]
    ])
    B = np.array([
        [0.4, 0.6],
        [0.8, 0.2],
        [0.5, 0.5]
    ])
    Q = np.array(['白', '黑', '白', '白', '黑'])
    alpha = np.zeros((len(Q), len(A)))
    # 计算alpha的值
    calc_alpha(pi, A, B, Q, alpha, common.convert_obs_seq_2_index)
    print("计算出来的最终alpha的值为:")
    print(alpha)

    # 计算一下序列Q出现的可能性到底有多大
    p = 0
    for i in alpha[-1]:
        p += i
    print("序列{}出现的可能性为:{}".format(Q, p))
