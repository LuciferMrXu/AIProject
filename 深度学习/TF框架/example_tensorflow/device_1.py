#encoding:utf-8

import tensorflow as tf
#GPU可以指定多个，CPU只有一个，本机是4核只能显示CPU:0
with tf.device('/cpu:0'):
    # 这个代码块中定义的操作，会在tf.device给定的设备上运行
    # 如果指定为GPU, 有一些操作，是不会再GPU上运行的（一定要注意）
    # 如果按照的tensorflow cpu版本，没法指定GPU运行环境的    
    a = tf.constant(1.0,dtype=tf.float32)
    b = tf.constant(2.0,dtype=tf.float32)
    c = tf.add(a,b)
#测试： allow_soft_placement=False
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    print(sess.run(c))