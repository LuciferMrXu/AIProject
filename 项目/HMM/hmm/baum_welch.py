# -- encoding:utf-8 --
"""
进行HMM的模型参数学习的算法(无监督的学习方式)
Create by ibf on 2018/10/25
"""

from wzx.hmm import forward_prob, gamma_prob, backward_prob, ksi_prob, common
import numpy as np


def baum_welch(pi, A, B, Q, max_iter=3, fetch_index_by_obs_seq=None):
    """
    根据传入的初始随机模型参数以及观测序列Q，使用Baum Welch算法进行HMM的模型求解
    :param pi:
    :param A:
    :param B:
    :param Q:
    :param max_iter:
    :param fetch_index_by_obs_seq:
    :return:
    """
    # 1. 获取索引值的方法初始化
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        # 如果在调用方法的时候没有给定序列转换索引的方式，那么就使用默认的转换方式
        # 默认的时候使用字符的ASCII码
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 2. 初始化相关参数
    T = len(Q)
    n = len(A)
    m = len(B[0])
    alpha = np.zeros((T, n))
    beta = np.zeros((T, n))
    gamma = np.zeros((T, n))
    ksi = np.zeros((T - 1, n, n))
    n_range = range(n)
    m_range = range(m)
    t_range = range(T)
    t_1_range = range(T - 1)

    # 3. 迭代更新
    for time in range(max_iter):
        # a. 在当前的pi、A、B的取值的情况下，根据序列Q求出概率值
        forward_prob.calc_alpha(pi, A, B, Q, alpha, fetch_index_by_obs_seq_f)
        backward_prob.calc_beta(pi, A, B, Q, beta, fetch_index_by_obs_seq_f)
        gamma_prob.calc_gamma(alpha, beta, gamma)
        ksi_prob.calc_ksi(alpha, beta, A, B, Q, ksi, fetch_index_by_obs_seq_f)

        # b. 计算pi、A、B的最优值
        # b1. 更新pi的值
        for i in n_range:
            pi[i] = gamma[0][i]

        # b2. 更新A的值
        for i in n_range:
            for j in n_range:
                # 1. 迭代计算分子和分母的值
                numerator = 0.0
                denominator = 0.0
                for t in t_1_range:
                    numerator += ksi[t][i][j]
                    denominator += gamma[t][i]
                # 2. 计算转移概率值
                if denominator == 0.0:
                    A[i][j] = 0.0
                else:
                    A[i][j] = numerator / denominator

        # b3. 更新状态和观测值之间的转移矩阵B
        for i in n_range:
            for j in m_range:
                # 1. 计算分子和分母值
                numerator = 0.0
                denominator = 0.0
                for t in t_range:
                    # 只有当t时刻的观测值为对应的j值的时候，那么才计算分子的值
                    if j == fetch_index_by_obs_seq_f(Q, t):
                        numerator += gamma[t][i]
                    denominator += gamma[t][i]
                # 2. 计算概率值
                if denominator == 0.0:
                    B[i][j] = 0.0
                else:
                    B[i][j] = numerator / denominator


if __name__ == '__main__':
    # 随机状态
    np.random.seed(28)
    pi = np.random.randint(1, 10, 3)
    pi = pi / np.sum(pi)
    A = np.random.randint(1, 10, (3, 3))
    A = A / np.sum(A, axis=1).reshape((-1, 1))
    B = np.random.randint(1, 10, (3, 2))
    B = B / np.sum(B, axis=1).reshape((-1, 1))

    # pi = np.array([0.2, 0.5, 0.3])
    # A = np.array([
    #     [0.5, 0.4, 0.1],
    #     [0.2, 0.2, 0.6],
    #     [0.2, 0.5, 0.3]
    # ])
    # B = np.array([
    #     [0.4, 0.6],
    #     [0.8, 0.2],
    #     [0.5, 0.5]
    # ])
    Q = np.array(['白', '黑', '白', '白', '黑'])
    Q1 = np.array(['白', '黑', '黑', '白', '黑'])
    print("初始的随机状态矩阵:")
    print("初始状态概率向量：")
    print(pi)
    print("\n初始的状态之间的转移概率矩阵：")
    print(A)
    print("\n初始的状态和观测值之间的转移概率矩阵：")
    print(B)

    # 计算结果
    baum_welch(pi, A, B, Q, fetch_index_by_obs_seq=common.convert_obs_seq_2_index)
    baum_welch(pi, A, B, Q1, fetch_index_by_obs_seq=common.convert_obs_seq_2_index)

    # 输出最终结果
    print("\n\n最终计算出来的状态矩阵:")
    print("状态概率向量：")
    print(pi)
    print("\n状态之间的转移概率矩阵：")
    print(A)
    print("\n状态和观测值之间的转移概率矩阵：")
    print(B)
