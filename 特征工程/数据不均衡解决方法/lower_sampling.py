# -- encoding:utf-8 --
"""
下采样(主要就是数据部分)
Create by ibf on 2018/7/28
"""

import numpy as np
import pandas as pd


def lower_sample_data(df, sample_number):
    """
    从给定的DataFrame对象df中抽取出sample number条样本数据，并将数据返回
    :param df:
    :param sample_number:
    :return:
    """
    # 1. 获取总的样本数目
    rows = len(df)

    # 2. 进行样本数目判断，如果样本数目小于需要抽取的sample number的数目，那么直接返回
    if rows <= sample_number:
        return None, df

    # 3. 随机生成数据对应的下标
    row_index = set()
    while len(row_index) != sample_number:
        index = np.random.randint(0, rows, 1)[0]
        row_index.add(index)

    # 4. 进行数据的抽取操作
    sample_df = df.iloc[list(row_index)]
    other_row_index = [i for i in range(rows) if i not in row_index]
    other_df = df.iloc[list(other_row_index)].reset_index(drop=True)

    # 5. 任何结果
    return other_df, sample_df


if __name__ == '__main__':
    # 给定随机数种子
    np.random.seed(28)

    # 1. 产生模拟数据
    category1 = np.random.randint(0, 10, [10000, 5]).astype(np.float)
    label1 = np.array([1] * 10000).reshape((-1, 1))
    data1 = np.concatenate((category1, label1), axis=1)
    category2 = np.random.randint(8, 18, [10, 5]).astype(np.float)
    label2 = np.array([0] * 10).reshape((-1, 1))
    data2 = np.concatenate((category2, label2), axis=1)

    # 2. 构建DataFrame
    name = ['A', 'B', 'C', 'D', 'E', 'Label']
    data = np.concatenate((data1, data2), axis=0)
    df = pd.DataFrame(data, columns=name)
    print(df.head(3))

    # 3. 查看一下数据的各个类别的样本数目
    print("各个类别的样本数目:")
    print(df.Label.value_counts())

    # 4. 获取大众类别的数据
    small_df = df[df.Label == 0.0]

    big_df = df[df.Label == 1.0]

    # 5. 进行采样的操作
    sample_number = 100
    big_df, sample_big_category_df = lower_sample_data(big_df, sample_number)
    print(big_df.shape)
    print(sample_big_category_df.shape)
    print(sample_big_category_df.head(2))

    # 5. 合并数据集
    train_df = pd.concat([small_df, sample_big_category_df], ignore_index=True)
    print(train_df.shape)
    print(train_df.head(5))
    print(train_df.Label.value_counts())

    # TODO: 后面基于train_df的数据就可以做模型训练了，训练完成后，继续抽样
