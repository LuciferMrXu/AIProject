import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

#https://github.com/balancap/SSD-Tensorflow/blob/master/notebooks/ssd_notebook.ipynb
# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
#tf.InteractiveSession()来构建会话的时候，我们可以先构建一个session然后再定义操作（operation）
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
#N是batch_size H高 W宽 C通道
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
#原始图像、图像label;box，anchor_box数据  
#image_pre是将img_input变为浮点数，resize白化后的图像数据；
#bbox_img四坐标数据
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
#predictions类别；localisations位置
#第一个步：设计vgg网络特征提取，并得到6中不同尺寸的特征图 
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.

ssd_anchors = ssd_net.anchors(net_shape)

# Main image processing routine. select_threshold=0.5 nms_th=0.15
#nms_th两个框IOU超过0.15的都丢弃，只留下置信度最高的一个；是其他框与置信度最高那个框的IOU大于阈值的框删除
def process_image(img, select_threshold=0.5, nms_threshold=.70, net_shape=(300, 300)):
    # Run SSD network.
    # Run SSD network. rimg.shape=[1,300,300,3] rpredictions.shape=[6,1,38,38,4,21] rlocalisations.shape=[6,1,38,38,4,4]
    #bbox_img是框图，localisations是位置，6    是6种尺寸featureMaps
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    #rlocalisations与rpredictions是anchor框与groudtruth框映射关系
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    #排序
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    #极大值抑制
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

# Test on some demo image and visualize output.
path = '../demo/'
image_names = sorted(os.listdir(path))

img = mpimg.imread(path + image_names[12])
rclasses, rscores, rbboxes =  process_image(img)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

'''
for it  in image_names:
    img = mpimg.imread(path + it)
    i+=1
    if i>4: break
    rclasses, rscores, rbboxes =  process_image(img)
    visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
'''