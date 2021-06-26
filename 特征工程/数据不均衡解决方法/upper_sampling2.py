# --encoding:utf-8 --
"""上采样操作"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 给定随机种子
np.random.seed(28)


def upper_sample_data(df, sample_number, label):
    """
      进行上采样
      df: DataFrame对象，进行上采样过程中的原始数据集
      sample_number： 需要采样的数量
      label: label名称，必须是字符串
    """
    # 1. 获取DataFrame的数量（行数和列数）
    df_column_size = df.columns.size
    df_row_size = len(df)
    random_range = range(df_column_size - 1)

    # 2. 进行抽样操作
    sample_df = pd.DataFrame(columns=df.columns)
    for i in range(sample_number):
        """进行抽样操作"""
        # a. 随机一个下标值
        index = np.random.randint(0, df_row_size, 1)[0]
        # b. 获取对应下标的数据
        item = df.iloc[index]
        # c. 获取对应数据的标签值
        label_value = item[label]
        # d. 删除标签值
        del item[label]
        # e. 进行随机偏移
        item = item + [(np.random.random() - 0.5) / 500 for j in random_range]
        # f. 给偏移之后的数据添加一个Label
        item[label] = label_value
        # g. 添加到最终的DataFrame中
        sample_df.loc[i] = item

    # 3. 返回最终结果
    return sample_df


if __name__ == '__main__':
    # 进行上采样操作

    # 1. 创建模拟数据
    category1 = np.random.randint(0, 10, [10000, 5]).astype(np.float)
    label1 = np.array([1] * 10000).reshape(-1, 1)
    data1 = np.concatenate((category1, label1), axis=1)
    category2 = np.random.randint(8, 18, [10, 5]).astype(np.float)
    label2 = np.array([0] * 10).reshape(-1, 1)
    data2 = np.concatenate((category2, label2), axis=1)

    name = ['A', 'B', 'C', 'D', 'E', 'Label']
    data = np.concatenate((data1, data2), axis=0)
    df = pd.DataFrame(data, columns=name)
    print(df.head())

    # 2. 查看各个类别的数据
    print("=" * 100)
    print(df.Label.value_counts())

    # 3. 获取小众类别的数据
    small_category = df[df.Label == 0.0]
    sample_number = 1000  # 需要抽样的数量
    sample_category_data = upper_sample_data(small_category, sample_number, label='Label')
    print("=" * 100)
    print(small_category.head())
    print("=" * 10)
    print(sample_category_data.head())

    # 4. 合并数据
    final_df = pd.concat([df, sample_category_data], ignore_index=True)
    print("=" * 100)
    print(final_df.head())
    print("=" * 100)
    print(final_df.Label.value_counts())

