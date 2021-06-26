# -- encoding:utf-8 --
"""
Create by ibf on 2018/10/25
"""

from wzx.hmm import common
import numpy as np


def viterbi(pi, A, B, Q, delta, fetch_index_by_obs_seq=None):
    """
    viterbi算法
    :param pi:
    :param A:
    :param B:
    :param Q:
    :param delta:
    :param fetch_index_by_obs_seq:
    :return:
    """
    # 1. 获取索引值的方法初始化
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        # 如果在调用方法的时候没有给定序列转换索引的方式，那么就使用默认的转换方式
        # 默认的时候使用字符的ASCII码
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 2. 获取参数
    T = len(Q)
    n = len(A)
    n_range = range(n)

    # 3. 定义临时参数
    # 用于存储上一个时刻的最优状态值，eg：pre_index[2][1]就表示t=3时刻处于状态i的最优上一个时刻的状态值为pre_index[2][1]
    pre_index = np.zeros((T, n), dtype=np.int32)

    # 4. 计算t=1时刻的对应的delta值
    for i in n_range:
        delta[0][i] = pi[i] * B[i][fetch_index_by_obs_seq_f(Q, 0)]

    # 5. 计算t=2,3,4...T时刻对应的delta值
    for t in range(1, T):
        for i in n_range:
            # a. 获取最大值
            max_delta = -1.0
            for j in n_range:
                tmp = delta[t - 1][j] * A[j][i]
                if tmp > max_delta:
                    max_delta = tmp
                    pre_index[t][i] = j
            delta[t][i] = max_delta * B[i][fetch_index_by_obs_seq_f(Q, t)]

    # 6. 解码操作
    decode = [-1 for i in range(T)]
    # 首先找到最后一个时刻对应的最优状态，也就是delta中概率值最大的那个状态
    max_delta_index = 0
    for i in range(1, n):
        if delta[T - 1][i] > delta[T - 1][max_delta_index]:
            max_delta_index = i
    decode[T - 1] = i
    # 然后基于最后一个时刻的最优状态，来找之前时刻的最有可能的状态
    for t in range(T - 2, -1, -1):
        max_delta_index = pre_index[t + 1][max_delta_index]
        decode[t] = max_delta_index

    return decode, delta


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
    delta = np.zeros((len(Q), len(A)))

    # 计算
    state_sq, _ = viterbi(pi, A, B, Q, delta, fetch_index_by_obs_seq=common.convert_obs_seq_2_index)

    print("最终结果:", end='')
    print(state_sq)
    state = ['盒子1', '盒子2', '盒子3']
    for i in state_sq:
        print(state[i], end='\t')
