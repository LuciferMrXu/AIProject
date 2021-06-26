# -- encoding:utf-8 --
"""
计算γ概率值
Create by ibf on 2018/10/18
"""

from wzx.hmm import forward_prob, backward_prob, common
import numpy as np


def calc_gamma(alpha, beta, gamma):
    """
    计算gamma值
    :param alpha:  给定的alpha前向概率值
    :param beta:  给定的beta后向概率值
    :param gamma:  需要计算的gamma矩阵
    :return:
    """
    # 1. 获取序列长度和状态数目
    T, n = np.shape(alpha)

    # 2. 更新gamma值
    tmp_prob = np.zeros(n)
    for t in range(T):
        # 1. 分别计算当前时刻t，状态为j的前向概率和后向概率的乘积
        for j in range(n):
            tmp_prob[j] = alpha[t][j] * beta[t][j]
        tmp_total_prob = np.sum(tmp_prob)

        # 2. 更新对应时刻t，状态为i的gamma概率值
        for i in range(n):
            gamma[t][i] = tmp_prob[i] / tmp_total_prob

    return gamma


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
    beta = np.zeros((len(Q), len(A)))
    alpha = np.zeros((len(Q), len(A)))
    gamma = np.zeros((len(Q), len(A)))
    # 计算alpha的值
    forward_prob.calc_alpha(pi, A, B, Q, alpha, common.convert_obs_seq_2_index)
    # 计算beta的值
    backward_prob.calc_beta(pi, A, B, Q, beta, common.convert_obs_seq_2_index)
    # 计算gamma值
    calc_gamma(alpha, beta, gamma)
    print("计算出来的最终alpha的值为:")
    print(alpha)
    print("计算出来的最终beta的值为:")
    print(beta)
    print("计算出来的最终gamma的值为:")
    print(gamma)
