import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image # 图像的读写显的库
from mpl_toolkits.mplot3d import Axes3D

from scipy import misc

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin # 计算点与点直接的最小距离，然后将最小距离的点组成一个key/value的键值对
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

# 加载图像（RGB图像(3维数组)或者灰度图像L（2维数组））
# 计算机中，对于图像来讲，是使用像素点来进行描述，每个像素点就是一个颜色
n_colors = 16
image = Image.open('./gray.png')
# print(type(image))
# print(image)

#将数据转换为RGB的百分比
image = np.array(image, dtype=np.float64) / 255
# print(image.shape)
# print(image)

#获取图片数据的维度/形状（长、宽、像素）
original_shape = tuple(image.shape)
shape_size = len(original_shape)
# print(shape_size)

# d设置为1是为了黑白图片的显示
w, h, d = original_shape[0], original_shape[1], 1


# 把每一个像素点当做一个样本，做聚类操作
image_v = np.reshape(image, (w * h, d))
# print(image_v.shape)
# print(image_v)

# 因为图像的像素太多，所以随机选择其中的10000个像素点
image_v_sample = shuffle(image_v, random_state=28)[:10000]    #shuffle()洗牌操作


# 压缩功能实现：数据存储的时候，使用更少的颜色空间来表示，使用更少的颜色来表示数据 ---> 就可以把每个像素点当做一个样本，做一个聚类 ---> 然后使用簇中心点坐标来替换原始数据的像素点值
k_means = KMeans(n_clusters=n_colors, random_state=28)
k_means.fit(image_v_sample)
labels = k_means.predict(image_v)

"""
重新构建一个图片数据(压缩图片数据)
codebook：各个类别的具体像素值集合(聚类中心/簇中心/类别中心)，中心点坐标
labels：原始图片各个像素点的类别集合（所对应的codebook中的坐标点的下标值）
w: 原始/新图片宽度
h：原始/新图片高度
"""
def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0 # 第几个原始图像的像素点
    for i in range(w):
        for j in range(h):
            # 获取原始数据像素点对应的类别中心点坐标
            # 再根据中心点坐标获取对应的像素值
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


plt.figure(u'K-Means算法压缩图片',facecolor='w')
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
km_image = recreate_image(k_means.cluster_centers_, labels, w, h)
km_image.shape = original_shape

plt.imshow(km_image, cmap=plt.cm.gray)


misc.imsave('datas/result.png', km_image)