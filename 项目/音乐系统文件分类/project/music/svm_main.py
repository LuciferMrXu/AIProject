# -- encoding:utf-8 --
"""
使用svm算法实现模型构建的相关API
Create by ibf on 2018/8/5
"""

import numpy as np
import pandas as pd
import random

import pymysql
from sqlalchemy import create_engine

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle
from sklearn.externals import joblib

from project.music import feature_process

random.seed(28)
np.random.seed(28)
music_feature_file_path = './data/music_feature.csv'
music_label_index_file_path = './data/music_index_label.csv'
default_model_out_f = './data/music_svm.model'

"""
大概需要些的算法：
1. 模型交叉验证的API，用于模型最优参数的选择
2. 具体的模型训练以及模型保存的API
3. 使用保存好的模型来预测数据，并产生结果的API
"""


def cross_validation(music_feature_csv_file_path=None, data_percentage=0.7):
    """
    交叉验证，用于选择模型的最优参数
    :param music_feature_csv_file_path: 训练数据的存储文件路径
    :param data_percentage:  给定使用多少数据用于模型选择
    :return:
    """
    # 1. 初始化文件路径
    if not music_feature_csv_file_path:
        music_feature_csv_file_path = music_feature_file_path
    # 2. 读取数据
    print("开始读取原始数据:{}".format(music_feature_csv_file_path))
    data = pd.read_csv(music_feature_csv_file_path, sep=',', header=None, encoding='utf-8')

    # 2. 抽取部分数据用于交叉验证
    sample_fact = 0.7
    if isinstance(data_percentage, float) and 0 < data_percentage < 1:
        sample_fact = data_percentage
    data = data.sample(frac=sample_fact, random_state=28)
    X = data.T[:-1].T
    Y = np.array(data.T[-1:]).reshape(-1)
    print(np.shape(X))
    print(np.shape(Y))

    # 3. 给定交叉验证的参数
    parameters = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        'degree': [2, 3, 4, 5, 6],
        'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'decision_function_shape': ['ovo', 'ovr']
    }

    # 4. 构建模型并训练
    print("开始进行模型参数选择....")
    model = GridSearchCV(estimator=SVC(random_state=28), param_grid=parameters, cv=3)
    model.fit(X, Y)

    # 5. 输出最优的模型参数
    print("最优参数:{}".format(model.best_params_))
    print("最优的模型Score值:{}".format(model.best_score_))


def fit_dump_model(music_feature_csv_file_path=None, train_percentage=0.7, model_out_f=None, fold=1):
    """
    进行fold次模型训练，最终将模型效果最好的那个模型保存到model_out_f文件中
    :param music_feature_csv_file_path:  训练数据存在的文件路径
    :param train_percentage:  训练数据占比
    :param model_out_f:  模型保存文件路径
    :param fold:  训练过程中，训练的次数
    :return:
    NOTE: 因为现在样本数据有点少，所以模型的衡量指标采用训练集和测试集的加权准确率作为衡量指标, eg: source = 0.35 * train_source + 0.65 * test_source
    """
    # 1. 变量初始化
    if not music_feature_csv_file_path:
        music_feature_csv_file_path = music_feature_file_path
    if not model_out_f:
        model_out_f = default_model_out_f

    # 2. 进行数据读取
    print("开始读取原始数据:{}".format(music_feature_csv_file_path))
    data = pd.read_csv(music_feature_csv_file_path, sep=',', header=None, encoding='utf-8')

    # 3. 开始进行循环处理
    max_train_source = None
    max_test_source = None
    max_source = None
    best_model = None
    flag = True
    for index in range(1, int(fold) + 1):
        # 3.1 开始进行数据的抽取、分割
        shuffle_data = shuffle(data).T
        X = shuffle_data[:-1].T
        Y = np.array(shuffle_data[-1:]).reshape(-1)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=train_percentage)

        # 3.2 模型训练
        svm = SVC(kernel='poly', C=1e-5, decision_function_shape='ovo', random_state=28, degree=2)
        # svm = SVC(kernel='rbf', C=1.0, gamma=1e-5, decision_function_shape='ovo', random_state=28, degree=2)
        # svm = SVC(kernel='poly', C=0.1, decision_function_shape='ovo', random_state=28, degree=3)
        svm.fit(x_train, y_train)

        # 3.3 获取准确率的值
        acc_train = svm.score(x_train, y_train)
        acc_test = svm.score(x_test, y_test)
        acc = 0.35 * acc_train + 0.65 * acc_test

        # 3.4 临时保存最优的模型
        if flag:
            max_source = acc
            max_test_source = acc_test
            max_train_source = acc_train
            best_model = svm
            flag = False
        elif max_source < acc:
            max_source = acc
            max_test_source = acc_test
            max_train_source = acc_train
            best_model = svm

        # 3.5 打印一下日志信息
        print("第%d次训练，测试集上准确率为:%.2f, 训练集上准确率为:%.2f，修正的准确率为:%.2f" % (index, acc_test, acc_train, acc))

    # 4. 输出最优模型的相关信息
    print("最优模型效果:测试集上准确率为:%.2f, 训练集上准确率为:%.2f，修正的准确率为:%.2f" % (max_test_source, max_train_source, max_source))
    print("最优模型为:")
    print(best_model)

    # 5. 模型存储
    joblib.dump(best_model, model_out_f)


def fetch_predict_label(X, model_file_path=None, label_index_file_path=None):
    """
    获取预测标签名称
    :param X: 特征矩阵
    :param model_file_path: 模型对象
    :param label_index_file_path:  标签id和name的映射文件
    :return:
    """
    # 1. 初始化相关参数
    if not model_file_path:
        model_file_path = default_model_out_f
    if not label_index_file_path:
        label_index_file_path = music_label_index_file_path

    # 2. 加载模型
    model = joblib.load(model_file_path)

    # 3. 加载标签的id和name的映射关系
    tag_index_2_name_dict = feature_process.fetch_index_2_label_dict(label_index_file_path)

    # 4. 做数据预测
    label_index = model.predict(X)

    # 5. 获取最终标签值
    result = np.array([])
    for index in label_index:
        result = np.append(result, tag_index_2_name_dict[index])

    # 6. 返回标签值
    return result


if __name__ == '__main__':
    flag = 3
    if flag == 1:
        """
        模型做一个交叉验证
        """
        cross_validation(data_percentage=0.9)
    elif flag == 2:
        """
        模型训练
        """
        fit_dump_model(train_percentage=0.9, fold=100)
    elif flag == 3:
        """
        直接控制台输出模型预测结果
        """
        _, X = feature_process.extract_music_feature('./data/test/*.mp3')
        print("X形状:{}".format(X.shape))
        label_names = fetch_predict_label(X)
        # label_names = fetch_predict_label(X, model_file_path='./data/music_model.pkl',
        #                                   label_index_file_path='./data/music_index_label2.csv')
        print(label_names)
    else:
        """
        使用训练好的模型，将所有音频数据的类别信息输入到数据库中
        TODO: 大家自己把这一段代码整理成为API的形式
        """
        name, X = feature_process.extract_music_feature('./data/test/*.mp3')
        label_names = fetch_predict_label(X)
        music_name_2_label = np.hstack((name.reshape((-1, 1)), label_names.reshape((-1, 1))))
        music_name_2_label_df = pd.DataFrame(music_name_2_label, columns=['name', 'label'])
        print(music_name_2_label_df)
        # 将结果输出到MySQL数据库中，使用pymysql和sqlalchemy库，安装方式：pip安装即可
        # a. 创建一个连接
        db_info = {
            'user': 'root',
            'password': 'root',
            'host': 'localhost',
            'database': 'test'}
        engine = create_engine('mysql+pymysql://%(user)s:%(password)s@%(host)s/%(database)s?charset=utf8' % db_info,
                               encoding='utf-8')

        # b. 数据输出
        music_name_2_label_df.to_sql(name='tb_music_tag', con=engine, if_exists='replace')
