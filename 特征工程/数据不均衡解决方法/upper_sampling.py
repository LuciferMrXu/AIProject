# -- encoding:utf-8 --
"""
上采样操作
Create by ibf on 2018/10/14
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression


def upper_sample_data(df, sample_number, label):
    # 1. 获取DataFrame的列数和行数
    df_column_size = df.columns.size
    df_row_size = len(df)

    # 2. 做抽样操作
    sample_df = pd.DataFrame(columns=df.columns)
    for i in range(sample_number):
        # 1. 随机一个样本下标值
        idx = np.random.randint(0, df_row_size, 1)[0]
        # 2. 获取下标对应的标签值
        item = df.iloc[idx]
        # 3. 获取原始数据的标准值
        label_value = item[label]
        # 4. 删除标签值
        del item[label]
        # 5. 对剩下的特征属性做一个偏移的操作
        item = item + [np.random.random() - 0.5 for j in range(df_column_size - 1)]
        # 6. 将标签值还原
        item[label] = label_value
        # 7. 将数据添加到DataFrame中
        sample_df.loc[i] = item
    return sample_df


if __name__ == '__main__':
    # 1. 产生一个模拟数据
    category1 = np.random.rand(10000, 2) * 10
    label1 = np.array([1] * 10000).reshape(-1, 1)
    data1 = np.concatenate((category1, label1), axis=1)
    category2 = np.random.rand(100, 2) * 3 - 2
    label2 = np.array([0] * 100).reshape(-1, 1)
    data2 = np.concatenate((category2, label2), axis=1)

    name = ['A', 'B', 'label']
    data = np.concatenate((data1, data2), axis=0)
    df = pd.DataFrame(data, columns=name)

    # 做一个上采样
    print("=" * 100)
    print(df.label.value_counts())

    # 获取小众样本的数据
    small_category = df[df.label == 0.0]
    sample_number = 2000
    sample_category_data = upper_sample_data(small_category, sample_number, label='label')
    print("=" * 100)
    print(sample_category_data.head(10))

    # 合并上采样产生的小众类别的数据和大众类别数据
    final_df = pd.concat([df, sample_category_data], ignore_index=True)
    print("=" * 100)
    print(final_df.label.value_counts())

    # # 1. 做数据划分操作
    df = final_df
    x = df.drop('label', axis=1)
    y = df['label']
    algo = LogisticRegression(fit_intercept=True)
    algo.fit(x, y)
    w1, w2 = algo.coef_[0]
    c = algo.intercept_
    print("模型参数:{}-{}-{}".format(w1, w2, c))

    # 查看原始数据
    plt.plot(category1[:, 0], category1[:, 1], 'ro', markersize=3)
    plt.plot(category2[:, 0], category2[:, 1], 'bo', markersize=3)
    plt.plot([-10, 10.0 * w1 / w2 - c / w2], [10, -10.0 * w1 / w2 - c / w2], 'g-')
    plt.show()
