# coding:utf-8

import tensorflow as tf

a = tf.Variable(3.0)
b = tf.constant(2.0)
c = tf.add(a,b)

#创建依赖型变量
w1 = tf.Variable(tf.random_normal([5],stddev=1.0,dtype=tf.float32),name='w1')
'''
关于变量：1 要初始化 有两种方式a.可以使用具体的数值初始化 b.可以使用初始化参数初始化
            (初始化函数可以将变量初始化为一个接近0而非0的矩阵，且符合某种分布）   
         2.变量可以被保存起来放在模型中
w11 = tf.Variable(1.0)
'''

input_x = tf.constant(2.0, dtype=tf.float32)
w2 = tf.Variable(w1.initialized_value()*input_x, name='w2')

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #1. 变量要初始化
    sess.run(tf.initialize_all_variables())
    #2. 变量和常量可以op
    #print(sess.run(c))
    print(c.eval())
    print(sess.run([w2,w1]))


#简化版
w1=tf.Variable(1.0,name='w1',dtype=tf.float32)
input_x=tf.constant(2.0)
y=tf.add(w1,input_x)
with tf.Session() as sess:
    #sess.run(tf.initialize_all_variables())    #初始化所有变量
    sess.run(tf.global_variables_initializer())
    print(sess.run(y))




