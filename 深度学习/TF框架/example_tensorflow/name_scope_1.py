#coding:utf-8

import tensorflow as tf
'''
名字作用域name_scope对tf.Variable定义的变量有用，对tf.get_variable定义的共享变量无用
'''
with tf.name_scope("name_scope_1"):
    a = tf.Variable(3,name='a')
    b = tf.get_variable("b",shape=[1],initializer=tf.random_uniform_initializer())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("{},{}".format(a.name, a.eval())) 
    print("{},{}".format(b.name, b.eval()))