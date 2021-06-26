# -- encoding:utf-8 --
"""
计算ksi的概率值
Create by ibf on 2018/10/18
"""

from wzx.hmm import forward_prob, backward_prob, common
import numpy as np


def calc_ksi(alpha, beta, A, B, Q, ksi, fetch_index_by_obs_seq=None):
    """
    计算ksi值
    :param alpha:  给定的alpha前向概率值
    :param beta:  给定的beta后向概率值
    :param A:  给定的模型状态之间的转移概率矩阵
    :param B:  给定的模型状态与观测值之间的转移概率矩阵
    :param Q:  观测值序列组成的一个向量/数组/集合
    :param ksi:  需要计算的ksi矩阵
    :param fetch_index_by_obs_seq:  根据序列参数返回对应索引值的函数，可以为None
    :return:
    """
    # 0. 获取索引值的方法初始化
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 1. 获取序列长度和状态数目
    T, n = np.shape(alpha)

    # 2. 更新ksi值
    tmp_prob = np.zeros((n, n))
    for t in range(T - 1):
        # 1. 计算分子的值
        obs_index = fetch_index_by_obs_seq_f(Q, t + 1)
        for i in range(n):
            for j in range(n):
                tmp_prob[i][j] = alpha[t][i] * A[i][j] * B[j][obs_index] * beta[t + 1][j]
        tmp_total_prob = np.sum(tmp_prob)

        # 2. 更新对应时刻t，状态为i,t+1时刻状态为j的ksi概率值
        for i in range(n):
            for j in range(n):
                ksi[t][i][j] = tmp_prob[i][j] / tmp_total_prob

    return ksi


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
    T = len(Q)
    n = len(A)
    beta = np.zeros((T, n))
    alpha = np.zeros((T, n))
    ksi = np.zeros((T - 1, n, n))
    # 计算alpha的值
    forward_prob.calc_alpha(pi, A, B, Q, alpha, common.convert_obs_seq_2_index)
    # 计算beta的值
    backward_prob.calc_beta(pi, A, B, Q, beta, common.convert_obs_seq_2_index)
    # 计算gamma值
    calc_ksi(alpha, beta, A, B, Q, ksi, common.convert_obs_seq_2_index)
    print("计算出来的最终alpha的值为:")
    print(alpha)
    print("计算出来的最终beta的值为:")
    print(beta)
    print("计算出来的最终ksi的值为:")
    print(ksi)
