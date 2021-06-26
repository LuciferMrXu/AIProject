# -- encoding:utf-8 --

import numpy as np
import tensorflow as tf

# # TODO: 大家自己画一下图代码实现
# # 1. 构造一个数据
# np.random.seed(28)
# N = 100
# x = np.linspace(0, 6, N) + np.random.normal(loc=0.0, scale=2, size=N)
# y = 14 * x - 7 + np.random.normal(loc=0.0, scale=5.0, size=N)
# # 将x和y设置成为矩阵
# x.shape = -1, 1
# y.shape = -1, 1
#
# # 2. 模型构建
# # 定义一个变量w和变量b
# # random_uniform：（random意思：随机产生数据， uniform：均匀分布的意思） ==> 意思：产生一个服从均匀分布的随机数列
# # shape: 产生多少数据/产生的数据格式是什么； minval：均匀分布中的可能出现的最小值，maxval: 均匀分布中可能出现的最大值
# w = tf.Variable(initial_value=tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0), name='w')
# b = tf.Variable(initial_value=tf.zeros([1]), name='b')
# # 构建一个预测值
# y_hat = w * x + b
#
# # 构建一个损失函数（回归问题用均方差构建损失函数）
# # 以MSE作为损失函数（预测值和实际值之间的平方和）
# loss = tf.reduce_mean(tf.square(y_hat - y), name='loss')
#
# # 以随机梯度下降的方式优化损失函数（迭代w）
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
# # 在优化的过程中，是让那个函数最小化
# train = optimizer.minimize(loss, name='train')
#
# # 全局变量更新
# init_op = tf.global_variables_initializer()
#
#
# # 运行
# def print_info(r_w, r_b, r_loss):
#     print("w={},b={},loss={}".format(r_w, r_b, r_loss))
#
#
# with tf.Session() as sess:
#     # 初始化
#     sess.run(init_op)
#
#     # 输出初始化的w、b、loss
#     r_w, r_b, r_loss = sess.run([w, b, loss])
#     print_info(r_w, r_b, r_loss)
#
#     # 进行训练(n次)
#     for step in range(500):
#         # 模型训练
#         sess.run(train)
#         # 输出训练后的w、b、loss
#         r_w, r_b, r_loss = sess.run([w, b, loss])
#         print_info(r_w, r_b, r_loss)

#简化版
# tf模拟SGD算法


n = 500
train_x = np.linspace(0,6,n) + np.random.normal(loc=0.0, scale=2, size=n)
train_y = 14 * train_x - 7 + np.random.normal(loc=0.0, scale=5.0, size=n)


# 构建一个样本的占位符信息
x_data = tf.placeholder(tf.float32, [10])
y_data = tf.placeholder(tf.float32, [10])

# 定义一个变量w和变量b
# random_uniform：（random意思：随机产生数据， uniform：均匀分布的意思） ==> 意思：产生一个服从均匀分布的随机数列
# shape: 产生多少数据/产生的数据格式是什么； minval：均匀分布中的可能出现的最小值，maxval: 均匀分布中可能出现的最大值
w = tf.Variable(initial_value=tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0), name='w')
b = tf.Variable(initial_value=tf.zeros([1]), name='b')
# 构建一个预测值
y_hat = w * x_data + b

# 构建一个损失函数
# 以MSE作为损失函数（预测值和实际值之间的平方和）
loss = tf.reduce_mean(tf.square(y_hat - y_data), name='loss')

global_step = tf.Variable(0, name='global_step', trainable=False)
# 以随机梯度下降的方式优化损失函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
# 在优化的过程中，是让那个函数最小化
train = optimizer.minimize(loss, name='train', global_step=global_step)


# 初始化
init = tf.global_variables_initializer()

# 总共训练迭代次数 一个epoch表示所有的样本训练都训练一次
'''
一个epoch需要循环num_batch次
'''
training_epochs = 100
# 批次数量 batch_size 一次训练的样本数，batch_num 一个epoch训练多少次
num_batch = int(n / 10)
# 训练迭代次数（打印信息）
display_step = 5

with tf.Session() as sess:
    # 变量初始化
    sess.run(init)

    for epoch in range(training_epochs):
        # 迭代训练
        avg_cost = 0
        # 打乱数据顺序 shuffle
        index = np.random.permutation(n)
        for i in range(num_batch):
            # 获取传入进行模型训练的数据对应索引
            xy_index = index[i * 10:(i + 1) * 10]
            # 构建传入的feed参数
            feeds = {x_data: train_x[xy_index], y_data: train_y[xy_index]}
            # 进行模型训练
            _, step, loss_v, w_v, b_v = sess.run([train, global_step, loss, w, b], feed_dict=feeds)
            # 可选：获取损失函数值
            avg_cost += sess.run(loss, feed_dict=feeds) / num_batch

        if epoch % display_step == 0:
            print('Step:{}, loss:{}, w:{}, b:{}'.format(step, loss_v, w_v, b_v))
            print("迭代次数: %03d/%03d 损失值: %.9f " % (epoch, training_epochs, avg_cost))






