# -- encoding:utf-8 --
"""
参考网站：
https://github.com/jiaaro/pydub
http://pydub.com/
https://github.com/jiaaro/pydub/blob/master/API.markdown
Create by ibf on 2018/10/27
"""

from pydub import AudioSegment
import numpy as np
import array

# 1. 读取数据
path = './data/20Hz-stero.wav'
# path = './data/我们的纪念.mp3'
# path = './data/我们的纪念.wav'
path = './data/童年.wav'
# path = './data/04.wav'
"""
file: 给定音频文件所在的磁盘路径
format=None: 给定文件的音频类型，参数可选；如果不给定的话，会使用文件的后缀作为读取的音频格式 
"""
song = AudioSegment.from_file(file=path)

# 2. 相关属性
size = len(song)
print("音频文件的长度信息(毫秒):{}".format(size))
channel = song.channels
print("音频的通道数目:{}".format(channel))
frame_rate = song.frame_rate
print("音频的抽样频率:{}".format(frame_rate))
sample_width = song.sample_width
print("音频的样本宽度:{}".format(sample_width))

# 3. 设置相关的特征属性
# 如果降低音乐通道会带来音频的损失，但是增加不会带来损失（频率和样本宽度也一样）
song = song.set_channels(channels=2)
song = song.set_frame_rate(frame_rate=44100)
song = song.set_sample_width(sample_width=2)

print("音频文件的长度信息(毫秒):{}".format(len(song)))
print("音频的通道数目:{}".format(song.channels))
print("音频的抽样频率:{}".format(song.frame_rate))
print("音频的样本宽度:{}".format(song.sample_width))

# 4. 音频文件的保存
song.export('./data/01.wav', format='wav')

# 5. 获取部分数据保存
# 获取前10秒的数据保存
song[:10000].export('./data/02.wav', format='wav')
# 获取最后10秒的数据保存
song[-10000:].export('./data/03.wav', format='wav')
# 获取中间的10秒数据保存
mid = len(song) // 2
song[mid - 5000: mid + 5000].export('./data/04.wav', format='wav')

# 6. 填充保存
# a. 将song对象转换为numpy的array对象
samples = np.array(song.get_array_of_samples()).reshape(-1)
# print(samples.shape)
# print(samples[:10])
# b. 填充操作
append_size = 60 * song.channels * song.frame_rate
# pad_width：给定在什么位置填充，以及填充多少个值；在samples数组的前面添加append_size个值，后面添加0个值
# mode：给定填充方式，constant表示常量填充，要填充的常量需要给定，通过参数constant_values给定
# constant_values: 给定填充的常量
samples = np.pad(samples, pad_width=(append_size, 0), mode='constant', constant_values=(0, 0))
# print(samples.shape)
# print(samples[:10])
# c. 将截取的数组转换为Segment对象
song = song._spawn(array.array(song.array_type, samples))
song.export('./data/05.wav', format='wav')

# 7. 对音乐文件的处理
# eg: 音调、音频增大/减小...
(song + 10).export('./data/07.wav', format='wav')
(song - 10).export('./data/08.wav', format='wav')
samples = np.array(song.get_array_of_samples()).reshape(-1)
samples = (samples * 2).astype(np.int)
song = song._spawn(array.array(song.array_type, samples))
song.export('./data/09.wav', format='wav')

# 8. 音乐的循环
(song * 2).export('./data/10.wav', format='wav')
