# -- encoding:utf-8 --
"""
特征处理的相关API
Create by ibf on 2018/8/4
"""

import os
import glob
from pydub import AudioSegment
from python_speech_features import mfcc
import numpy as np
import pandas as pd
from scipy.io import wavfile

music_audio_regex_dir = './data/music/*.mp3'
music_info_csv_file_path = './data/music_info.csv'
music_feature_file_path = './data/music_feature.csv'
music_label_index_file_path = './data/music_index_label.csv'


def extract(file, file_format):
    """
    从指定的file文件中提取对应的mfcc特征信息
    :param file:  对应的文件夹路径
    :param file_format: 文件格式
    :return:
    """
    result = []
    # 1. 如果文件不是wav格式的，那么将其转换为wav格式
    is_tmp_file = False
    if file_format != 'wav':
        try:
            # 数据读取
            song = AudioSegment.from_file(file, format=file_format)
            # 获得输出的文件路径
            wav_file_path = file.replace(file_format, 'wav')
            # 数据输出
            song.export(out_f=wav_file_path, format='wav')
            is_tmp_file = True
        except Exception as e:
            result = []
            print("Error:. " + file + " to wav format file throw exception. msg:=", end='')
            print(e)
    else:
        wav_file_path = file

    # 2. 进行mfcc数据提取操作
    try:
        # 读取wav格式数据
        (rate, data) = wavfile.read(wav_file_path)

        # 提取mfcc特征
        mfcc_feat = mfcc(data, rate, numcep=13, nfft=2048)

        # 由于file文件对应的音频数据是最原始的音频数据，大小、通道、采样频率等指标都是不一样的，所以说最终mfcc_feat得到的结果大小是不一致的, 一般的格式为: [?, 13]； 也就是说mfcc_feat是一个未知行数，但是列数为13的二维矩阵。
        # 这样就导致每个样本(音频数据)的特征信息数目是不一致的，所以需要在这里对这个内容进行处理，让特征维度变的一致
        # 解决方案一：使用AudioSegment中的相关API，对音频做一个预处理，让所有音频数据的长度一致、各种特征信息也一致，这样可以保证提取出来的mfcc特征数目一致。
        # 解决方案二：在现在的mfcc的指标上，提取出更高层次的特征: 即均值和协方差
        # 解决方案二实现代码：
        # 1. 矩阵的转置
        mm = np.transpose(mfcc_feat)
        # 2. 求每行(求13个特征属性中各个特征属性的均值)
        mf = np.mean(mm, axis=1)
        # 3. 求13个特征属性之间的协方差矩阵，以协方差矩阵作为更高层次的特征
        cf = np.cov(mm)
        # 4. 将均值和协方差合并
        # 最终结果维度: 13 +13 + 12 + 11 + 10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 = 104
        result = mf
        for i in range(mm.shape[0]):
            # 获取协方差矩阵上对角线上的内容，添加到result中
            result = np.append(result, np.diag(cf, i))

        # 5. 结果返回
        return result
    except Exception as e:
        result = []
        print(e)
    finally:
        # 如果是临时文件，删除临时文件
        if is_tmp_file:
            os.remove(wav_file_path)
    return result


def extract_label():
    """
    提取标签数据，返回对的的Dict类型的数据
    :return:
    """
    # 1. 读取标签数据，得到DataFrame对象
    df = pd.read_csv(music_info_csv_file_path)
    # 2. 将DataFrame转换为Dict的对象，以name作为key，以tag作为value值
    name_label_list = np.array(df).tolist()
    name_label_dict = dict(map(lambda t: (t[0].lower(), t[1]), name_label_list))
    labels = set(name_label_dict.values())
    label_index_dict = dict(zip(labels, np.arange(len(labels))))
    # 3. 返回结果
    return name_label_dict, label_index_dict


def extract_and_export_all_music_feature_and_label():
    """
    提取所有的music的特征属性数据
    :return:
    """
    # 1. 读取csv文件格式数据得到音乐名称对应的音乐类型所组成的dict类型的数据
    name_label_dict, label_index_dict = extract_label()

    # 2. 获取所有音频文件对应的路径
    music_files = glob.glob(music_audio_regex_dir)
    music_files.sort()

    # 3. 遍历所有音频文件得到特征数据
    flag = True
    music_features = np.array([])
    for file in music_files:
        print("开始处理文件:{}".format(file))

        # a. 提取文件名称
        music_name_format = file.split("\\")[-1].split("-")[-1].split('.')
        music_name = '.'.join(music_name_format[0:-1]).strip().lower()
        music_format = music_name_format[-1].strip().lower()

        # b. 判断musicname对应的label是否存在，如果存在，那么就获取音频文件对应的mfcc值，如果不存在，那么直接过滤掉
        if music_name in name_label_dict:
            # c. 获取该文件对应的标签label
            label_index = label_index_dict[name_label_dict[music_name]]
            # d. 获取音频文件对应的mfcc特征属性，ff是一个一维的数组
            ff = extract(file, music_format)
            if len(ff) == 0:
                continue
            # e. 将标签添加到特征属性之后
            ff = np.append(ff, label_index)
            # b. 将当前音频的信息追加到数组中
            if flag:
                music_features = ff
                flag = False
            else:
                music_features = np.vstack([music_features, ff])
        else:
            print("没法处理:" + file + "; 原因：找不到对应的label标签!!!")

    # 4. 特征数据存储
    label_index_list = []
    for label in label_index_dict:
        label_index_list.append([label, label_index_dict[label]])
    pd.DataFrame(label_index_list).to_csv(music_label_index_file_path, header=None, index=False, encoding='utf-8')
    pd.DataFrame(music_features).to_csv(music_feature_file_path, header=None, index=False, encoding='utf-8')

    # 5. 直接返回
    return music_features


def extract_music_feature(audio_regex_file_path):
    """
    提取给定字符串对应的音频数据对应的mfcc格式的特征属性矩阵
    :param audio_regex_file_path:
    :return:
    """
    # 1. 获取文件夹下的所有音乐文件
    all_music_files = glob.glob(audio_regex_file_path)
    all_music_files.sort()
    # 2. 最终返回的mfcc矩阵
    flag = True
    music_names = np.array([])
    music_features = np.array([])
    for file in all_music_files:
        print("开始处理文件:{}".format(file))

        # a. 提取文件名称
        music_name_and_format = file.split("\\")[-1].split("-")
        music_name_and_format2 = music_name_and_format[-1].split('.')
        music_name = '-'.join(music_name_and_format[:-1]) + '-' + '.'.join(music_name_and_format2[:-1])
        music_format = music_name_and_format2[-1].strip().lower()

        # b. 获取音频文件对应的mfcc特征属性，ff是一个一维的数组
        ff = extract(file, music_format)
        if len(ff) == 0:
            print("提取" + file + "音频文件对应的特征数据出现异常!!!")
            continue

        # c. 将当前音频的信息追加到数组中
        if flag:
            music_features = ff
            flag = False
        else:
            music_features = np.vstack([music_features, ff])

        # 添加文件名称
        music_names = np.append(music_names, music_name)
    return music_names, music_features


def fetch_index_2_label_dict(file_path=None):
    """
    获取类别id对应类别名称组成的字典对象
    :param file_path: 给定文件路径
    :return:
    """
    # 0. 初始化文件
    if file_path is None:
        file_path = music_label_index_file_path
    # 1. 读取数据
    df = pd.read_csv(file_path, encoding='utf-8', header=None)
    # 2. 顺序交换形成dict对象
    label_index_list = np.array(df)
    index_label_dict = dict(map(lambda t: (t[1], t[0]), label_index_list))
    # 3. 返回
    return index_label_dict


if __name__ == '__main__':
    # 1. 提取所有的训练数据
    # extract_and_export_all_music_feature_and_label()

    # 2. 提取音乐文件对应的mfcc的特征矩阵信息，主要用于模型训练好后的：模型的预测
    print("*" * 100)
    _, mfcc_features = extract_music_feature('./data/test/*.mp3')
    print(mfcc_features)
    print(mfcc_features.shape)
