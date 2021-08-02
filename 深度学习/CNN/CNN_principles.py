'''
    卷积原理
'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# 多个卷积核进行卷积
def conv(image, filters, stride=1):
    conv_results = [conv_(image, f, stride) for f in filters]

    return np.array(conv_results)

# 一个卷积核的卷积操作
def conv_(image, filter, stride=1):
    height = image.shape[0] - filter.shape[0] + 1
    width = image.shape[1] - filter.shape[1] + 1

    # 初始化卷积输出
    conv_result = np.zeros(shape=(height // stride, width // stride))

    for h in range(0, height, stride):
        for w in range(0, width, stride):
            # 滑动窗
            window = image[h: h + filter.shape[0], w: w + filter.shape[1], :]
            conv_result[h][w] = np.sum(np.multiply(window, filter))
            # np.max(window) => max pooling最大池化
            # np.mean(window) => mean pooling平均池化

    return conv_result


if __name__ == '__main__':

    data = Image.open(os.path.join(BASE_DIR,'data/dog.jpg'))

    dog = np.array(data)
    ic(dog.shape)
    plt.imshow(dog)
    plt.show()

    # 变成一个信道的图形
    singledog = data.convert('L')
    plt.imshow(np.array(singledog))
    plt.show()

    # 自定义卷积核
    filter_ = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ])

    filter_2 = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])

    # 经过一轮卷积操作的输出
    dog_conv = conv_(dog, filter_)
    plt.imshow(dog_conv)
    plt.show()

    dog_convs = conv(dog, [filter_, filter_2])
    plt.imshow(dog_convs[1,:,:])
    plt.show()

    ic(dog_convs.shape)

    # 把特征从多维拉成一维
    flatten = dog_convs.reshape(1, -1)

    outputs = np.matmul(flatten, np.random.random(size=(flatten.shape[1], 5)))

    print(outputs)


