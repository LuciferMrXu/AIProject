# -- encoding:utf-8 --
"""
将概率的计算从累乘变成累加的操作
Create by ibf on 2018/11/2
"""

import numpy as np
import common


def calc_alpha(pi, A, B, Q, alpha, fetch_index_by_obs_seq=None):
    """
    计算前向概率α的值, 该概率是经过对数化后的值
    :param pi:  给定的模型的初始状态概率向量， pi也是经过对数化之后的概率值
    :param A:  给定的模型状态之间的转移概率矩阵， A也是经过对数化之后的概率值
    :param B:  给定的模型状态与观测值之间的转移概率矩阵， B也是经过对数化之后的概率值
    :param Q:  观测值序列组成的一个向量/数组/集合
    :param alpha:  是需要更新的一个前向概率矩阵， 计算的概率是经过对数化之后的概率值
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
        alpha[0][i] = pi[i] + B[i][fetch_index_by_obs_seq_f(Q, 0)]

    # 4. 更新t=1到t=T-1时刻对应的前向概率的值
    tmp = np.zeros(n)
    for t in range(1, T):
        for i in n_range:
            # a. 获取上一个时刻到当前时刻的状态转移概率
            for j in n_range:
                tmp[j] = alpha[t - 1][j] + A[j][i]

            # b. 更新当前时刻t的对应前向概率值
            alpha[t][i] = common.log_sum_exp(tmp)
            alpha[t][i] += B[i][fetch_index_by_obs_seq_f(Q, t)]

    # 5. 返回最终的更新值
    return alpha


def calc_beta(pi, A, B, Q, beta, fetch_index_by_obs_seq=None):
    """
    计算后向概率α的值, 该概率是经过对数化后的值
    :param pi:  给定的模型的初始状态概率向量， pi也是经过对数化之后的概率值
    :param A:  给定的模型状态之间的转移概率矩阵， A也是经过对数化之后的概率值
    :param B:  给定的模型状态与观测值之间的转移概率矩阵， B也是经过对数化之后的概率值
    :param Q:  观测值序列组成的一个向量/数组/集合
    :param beta:  是需要更新的一个后向概率矩阵， 计算的概率是经过对数化之后的概率值
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
        beta[T - 1][i] = 0

    # 4. 更新t=T-2到t=0时刻对应的前向概率的值
    tmp = np.zeros(n)
    for t in range(T - 2, -1, -1):
        for i in n_range:
            # a. 获取到下一个时刻的概率值
            obs_index = fetch_index_by_obs_seq_f(Q, t + 1)
            for j in n_range:
                tmp[j] = A[i][j] + beta[t + 1][j] + B[j][obs_index]

            # b. 更新当前时刻t的对应后向概率值
            beta[t][i] = common.log_sum_exp(tmp)

    # 5. 返回最终的更新值
    return beta


def calc_gamma(alpha, beta, gamma):
    """
    计算gamma值, 该概率是经过对数化后的值
    :param alpha:  给定的alpha前向概率值，alpha也是经过对数化之后的概率值
    :param beta:  给定的beta后向概率值，beta也是经过对数化之后的概率值
    :param gamma:  需要计算的gamma矩阵, 该概率是经过对数化后的值
    :return:
    """
    # 1. 获取序列长度和状态数目
    T, n = np.shape(alpha)

    # 2. 更新gamma值
    tmp_prob = np.zeros(n)
    for t in range(T):
        # 1. 分别计算当前时刻t，状态为j的前向概率和后向概率的乘积
        for j in range(n):
            tmp_prob[j] = alpha[t][j] + beta[t][j]
        tmp_total_prob = common.log_sum_exp(tmp_prob)

        # 2. 更新对应时刻t，状态为i的gamma概率值
        for i in range(n):
            gamma[t][i] = tmp_prob[i] - tmp_total_prob

    return gamma


def calc_ksi(alpha, beta, A, B, Q, ksi, fetch_index_by_obs_seq=None):
    """
    计算ksi值, 该概率是经过对数化后的值
    :param alpha:  给定的alpha前向概率值, 该概率是经过对数化后的值
    :param beta:  给定的beta后向概率值, 该概率是经过对数化后的值
    :param A:  给定的模型状态之间的转移概率矩阵, 该概率是经过对数化后的值
    :param B:  给定的模型状态与观测值之间的转移概率矩阵, 该概率是经过对数化后的值
    :param Q:  观测值序列组成的一个向量/数组/集合
    :param ksi:  需要计算的ksi矩阵, 该概率是经过对数化后的值
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
                tmp_prob[i][j] = alpha[t][i] + A[i][j] + B[j][obs_index] + beta[t + 1][j]
        tmp_total_prob = common.log_sum_exp(tmp_prob.flat)

        # 2. 更新对应时刻t，状态为i,t+1时刻状态为j的ksi概率值
        for i in range(n):
            for j in range(n):
                ksi[t][i][j] = tmp_prob[i][j] - tmp_total_prob

    return ksi


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
        calc_alpha(pi, A, B, Q, alpha, fetch_index_by_obs_seq_f)
        calc_beta(pi, A, B, Q, beta, fetch_index_by_obs_seq_f)
        calc_gamma(alpha, beta, gamma)
        calc_ksi(alpha, beta, A, B, Q, ksi, fetch_index_by_obs_seq_f)

        # b. 计算pi、A、B的最优值
        # b1. 更新pi的值
        for i in n_range:
            pi[i] = gamma[0][i]

        # b2. 更新A的值
        numerator = np.zeros(T - 1)
        denominator = np.zeros(T - 1)
        for i in n_range:
            for j in n_range:
                # 1. 迭代计算分子和分母的值
                for t in t_1_range:
                    numerator[t] = ksi[t][i][j]
                    denominator[t] = gamma[t][i]
                # 2. 计算转移概率值
                A[i][j] = common.log_sum_exp(numerator) - common.log_sum_exp(denominator)

        # b3. 更新状态和观测值之间的转移矩阵B
        for i in n_range:
            for j in m_range:
                # 1. 计算分子和分母值
                numerator = np.zeros(T)
                denominator = np.zeros(T)
                number = 0
                for t in t_range:
                    # 只有当t时刻的观测值为对应的j值的时候，那么才计算分子的值
                    if j == fetch_index_by_obs_seq_f(Q, t):
                        numerator[number] = gamma[t][i]
                        number += 1
                    denominator[t] = gamma[t][i]
                # 2. 计算概率值
                if number == 0.0:
                    B[i][j] = float(-2 ** 31)
                else:
                    B[i][j] = common.log_sum_exp(numerator[:number]) - common.log_sum_exp(denominator)


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
        delta[0][i] = pi[i] + B[i][fetch_index_by_obs_seq_f(Q, 0)]

    # 5. 计算t=2,3,4...T时刻对应的delta值
    for t in range(1, T):
        for i in n_range:
            # a. 获取最大值
            max_delta = delta[t - 1][0] + A[0][i]
            for j in range(1, n):
                tmp = delta[t - 1][j] + A[j][i]
                if tmp > max_delta:
                    max_delta = tmp
                    pre_index[t][i] = j
            delta[t][i] = max_delta + B[i][fetch_index_by_obs_seq_f(Q, t)]

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
