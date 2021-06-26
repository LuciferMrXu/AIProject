# -- encoding:utf-8 --
"""
提取音频文件的MFCC的特征数据
Create by ibf on 2018/10/27
"""

from scipy.io import wavfile
from python_speech_features import mfcc

# 1. 读取wav格式的数据
path = './data/pydub/1.wav'
(rate, data) = wavfile.read(path)
print("音频文件的抽样频率:{}".format(rate))
print("音频文件的数据大小:{}".format(data.shape))
print(data[44100:44110])

print("*" * 100)
# 2. 提取MFCC的值
"""
signal: 给定音频数据的数据，是一个数组的形式
samplerate: 给定的音频数据的抽样频率
numcep：在MFCC进行倒谱操作的时候，给定将一个时间段分割为多少个空间，也可以认为是使用多少个特征信息来进行体现音频数据的
nfft：在傅里叶变换的过程中，参数值
"""
mfcc_feat = mfcc(signal=data, samplerate=rate, numcep=26, nfft=2048)
print(type(mfcc_feat))
print(mfcc_feat.shape)
print(mfcc_feat)
n_sample = mfcc_feat.reshape(-1)
print("最终的特征大小:{}".format(n_sample.shape))