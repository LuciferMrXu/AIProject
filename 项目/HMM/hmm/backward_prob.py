# -- encoding:utf-8 --
"""
后向算法
Create by ibf on 2018/10/18
"""

from wzx.hmm import common
import numpy as np


def calc_beta(pi, A, B, Q, beta, fetch_index_by_obs_seq=None):
    """
    计算后向概率α的值
    :param pi:  给定的模型的初始状态概率向量
    :param A:  给定的模型状态之间的转移概率矩阵
    :param B:  给定的模型状态与观测值之间的转移概率矩阵
    :param Q:  观测值序列组成的一个向量/数组/集合
    :param beta:  是需要更新的一个后向概率矩阵
    :param fetch_index_by_obs_seq:  根据序列参数返回对应索引值的函数，可以为None
    :return:  更新完成后，返回beta参数
    """
    # 1. 获取索引值的方法初始化
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 2. 定义状态相关信息
    n = np.shape(A)[0]
    T = np.shape(Q)[0]
    n_range = range(n)

    # 3. 更新t=T-1时刻对应的前向概率的值
    for i in n_range:
        beta[T - 1][i] = 1

    # 4. 更新t=T-2到t=0时刻对应的前向概率的值
    for t in range(T - 2, -1, -1):
        for i in n_range:
            # a. 获取到下一个时刻的概率值
            tmp_prob = 0.0
            obs_index = fetch_index_by_obs_seq_f(Q, t + 1)
            for j in n_range:
                tmp_prob += A[i][j] * beta[t + 1][j] * B[j][obs_index]

            # b. 更新当前时刻t的对应后向概率值
            beta[t][i] = tmp_prob

    # 5. 返回最终的更新值
    return beta


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
    # 计算beta的值
    calc_beta(pi, A, B, Q, beta, common.convert_obs_seq_2_index)
    print("计算出来的最终beta的值为:")
    print(beta)

    # 计算一下序列Q出现的可能性到底有多大
    p = 0
    for i in range(len(A)):
        p += pi[i] * B[i][common.convert_obs_seq_2_index(Q, 0)] * beta[0][i]
    print("序列{}出现的可能性为:{}".format(Q, p))
