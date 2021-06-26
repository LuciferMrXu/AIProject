# -- encoding:utf-8 --
"""
Create by ibf on 2018/10/30
"""

import math
import numpy as np

import hmm_learn

infinite = float(-2 ** 31)


def log_normalize(a):
    sum = math.log(np.sum(a))
    for i in range(len(a)):
        if a[i] == 0:
            a[i] = infinite
        else:
            a[i] = math.log(a[i]) - sum


def fit(train_file_path, mode='r', encoding='utf-8'):
    """
    基于传入的分好词的文本数据文件进行HMM模型参数的学习
    :param train_file_path:
    :param mode:
    :param encoding:
    :return:
    """
    # 1. 加载数据
    with open(train_file_path, mode=mode, encoding=encoding) as reader:
        # 读取所有数据,因为数据文件是BOM格式数据，在该文件格式中，第一个字符是不可见字符串，用于文件描述信息
        sentence = reader.read()[1:]

    # 2. 初始化相关的模型参数
    # 4表示隐状态的数目为4个，65536表示中文文字总数为65536个
    pi = np.zeros(4)
    A = np.zeros((4, 4))
    B = np.zeros((4, 65536))

    # 3. 模型训练
    # 隐状态：0B/1M/2E/3S O表示开始字符，1表示中间字符，2表示结尾字符，3表示单独字符
    # Begin, Middle, End, Single ---> 也就是表示位置信息
    tokens = sentence.split(' ')
    last_state = 2
    for k, token in enumerate(tokens):
        # 对于当前单词去掉前后空格
        token = token.strip()
        # 获取当前单词的长度
        length = len(token)

        # 如果长度小于等于0，那么表示token是空格，不需要考虑
        if length <= 0:
            continue

        # 如果长度等于1，那么特殊处理，认为当前单词是一个独立的字符
        if length == 1:
            pi[3] += 1
            A[last_state][3] += 1
            # ord函数的作用就是将字符转换为ACSII码
            B[3][ord(token[0])] += 1
            last_state = 3
            continue

        # 如果单词长度大于1，那么表示存在中间的单词以及结尾的单词，也就是这个词语不是单独存在的
        # a. 更新初始化状态向量
        pi[0] += 1   # 开头的单词数量加一
        #pi[2] += 1   # 结尾的单词数量加一
        #pi[1] += (n-2)  # 中间的单词数目累加

        # b. 更新状态转移矩阵
        A[last_state][0] += 1
        last_state = 2
        if length == 2:
            A[0][2] += 1
        else:
            A[0][1] += 1
            A[1][2] += 1
            A[1][1] += (length - 3)

        # c. 更新状态到观测值的转移矩阵
        B[0][ord(token[0])] += 1
        B[2][ord(token[-1])] += 1
        for i in range(1, length - 1):
            B[1][ord(token[i])] += 1

    # 4. 将统计的数量进行计算，得到概率值
    log_normalize(pi)  # 为了防止某些概率值较低，这里使用对数化概率的形式
    for i in range(4):
        log_normalize(A[i])
        log_normalize(B[i])

    return pi, A, B


# 模型保存成二进制文件 pickle.dump()
def dump(pi, A, B):
    """
    模型保存
    :param pi:
    :param A:
    :param B:
    :return:
    """
    n, m = np.shape(B)

    # 1. pi的输出
    with open('pi.txt', 'w') as f_pi:
        f_pi.write(str(n))
        f_pi.write('\n')
        f_pi.write(' '.join(map(str, pi)))
    # 2. A的输出
    with open('A.txt', 'w') as f_a:
        f_a.write(str(n))
        f_a.write('\n')
        for a in A:
            f_a.write(' '.join(map(str, a)))
            f_a.write('\n')
    # 3. B的输出
    with open('B.txt', 'w') as f_b:
        f_b.write(str(n))
        f_b.write('\n')
        f_b.write(str(m))
        f_b.write('\n')
        for b in B:
            f_b.write(' '.join(map(str, b)))
            f_b.write('\n')


def load():
    with open('pi.txt', 'r', encoding='utf-8') as f_pi:
        f_pi.readline()  # 第一行不需要
        line = f_pi.readline()
        pi = list(map(float, line.strip().split(' ')))

    with open('A.txt', 'r', encoding='utf-8') as f_a:
        n = int(f_a.readline())
        A = np.zeros((n, n))
        i = 0
        for line in f_a:
            j = 0
            for v in map(float, line.strip().split(' ')):
                A[i][j] = v
                j += 1
            i += 1

    with open('B.txt', 'r', encoding='utf-8') as f_b:
        n = int(f_b.readline())
        m = int(f_b.readline())
        B = np.zeros((n, m))
        i = 0
        for line in f_b:
            j = 0
            for v in map(float, line.strip().split(' ')):
                B[i][j] = v
                j += 1
            i += 1

    return pi, A, B

# 基于decode隐状态对data数据做分词操作
def segment(data, decode):
    T = len(data)
    t = 0
    while t < T:
        if decode[t] == 0 or decode[t] == 1:
            # t时刻所对应的文字是开头的字符或者中间字符
            j = t + 1
            while j < T:
                if decode[j] == 2:
                    break
                j += 1
            # t和j之间的就是一个单词
            print(data[t:j + 1], end="|")
            t = j
        elif decode[t] == 3 or decode[t] == 2:
            # 如果是单个字符或者给结尾的字符，那么直接一个字符串成为一个单词
            print(data[t:t + 1], end='|')
        else:
            print("ERROR!!")
        t += 1


if __name__ == '__main__':
    flag = False
    if flag:
        pi, A, B = fit("pku_training.utf8")
        dump(pi, A, B)
    else:
        # a. 加载参数
        pi, A, B = load()
        # b. viterbi算法分词
        with open('novel.txt', 'r', encoding='utf-8') as reader:
            sentence = reader.read()[1:]
        delta = np.zeros((len(sentence), len(A)))
        decode, _ = hmm_learn.viterbi(pi, A, B, sentence, delta)
        # c. 对获取得到的隐状态做一个解码的操作
        segment(sentence, decode)
