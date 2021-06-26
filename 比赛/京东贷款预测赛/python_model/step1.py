# -- encoding:utf-8 --
"""
查看数据的相关信息
Create by ibf on 2018/8/3
"""

import pandas as pd
import glob


def show_info(file_path):
    dir1 = glob.glob(file_path)
    for path1 in dir1:
        print(path1)
        print('File path : %s' % path1, end='\n')
        df = pd.read_csv(path1, encoding='utf-8')
        print("该文件中，唯一用户id数目为:{}".format(len(df.uid.unique())))
        print("该文件中，样本数目为:{}".format(df.shape[0]))
        print("该文件中，各个特征属性为空的样本数目为:\n{}".format(df.isnull().sum()))
        print("显示前5条数据:")
        print(df.head(5))
        print("该文件夹中，各个特征属性的相关信息为：")
        print(df.describe().T)
        print("\n")


def merge_data(dir_file_path, save_file_path):
    """
    合并文件数据成为DataFrame并输出到磁盘形成csv格式的文件
    :return:
    """
    dir1 = glob.glob(dir_file_path)
    for iid, path in enumerate(dir1):
        df1 = pd.read_csv(path)
        if iid == 0:
            df_all = df1
        else:
            df_all = pd.merge(df_all, df1, how='outer', on='uid')
    print("df_all:\n", df_all.shape)
    df_all.to_csv(save_file_path, header=True, index=None)


if __name__ == '__main__':
    flag = False
    if flag:
        show_info("../data/csv/*csv")
    else:
        merge_data('../data/feature/*csv', '../data/data_full.csv')
