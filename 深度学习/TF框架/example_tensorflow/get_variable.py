#encoding:utf-8

import tensorflow as tf

#使用非共享变量tf.variable
def fun1(x):
    w1 = tf.Variable(1)
    b1 = tf.Variable(1)
    r1 = w1 * x + b1

    w2 = tf.Variable(2)
    b2 = tf.Variable(2)
    r2 = w2 * r1 + b2

    return r1,r2
r = fun1(1)


#定义共享变量get_varialbe
def fun2(x):
    w1 = tf.get_variable('w1',shape=[1],initializer=tf.random_normal_initializer())
    b1 = tf.get_variable('b1',shape=[1],initializer=tf.random_normal_initializer(1.0))
    r = w1*x + b1
    return r

def func(x):
    #tf.AUTO_REUSE有自动屏蔽上层reuse的作用，这里reuse=False
    with tf.variable_scope('op1'):
        r1 = fun2(x)    #w1:op1/w1:0 b1:op1/b1:0
    with tf.variable_scope('op2',reuse=tf.AUTO_REUSE):
        r2 = fun2(r1)   #w1:op2/w1:0 b1:op2/b1:0
    return r1, r2
#链路分析：
with tf.variable_scope("get_variable_1"):
    r1 = func(1)
#链路分析：如果get_variable_1 reuse=True 被重用;以下没有被重用
with tf.variable_scope("get_variable_2",reuse=False):  #
    r2 = func(2)

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    sess.run(tf.initialize_all_variables())
    print("variable is:")
    print(sess.run(r))
    print("get_variable is :")
    print(sess.run([r1,r2]))
    #print(sess.run(r2))

#简化版
'''
1、get_variable定义变量只能初始化一次，而且是第一次定义的时候初始化
2、若要变量共享，一定要设置共享标志reuse为True
3、一旦将共享标志设置为true，就不能再初始化任何get_variable定义的变量了，只能找已经初始化的变量
4、一种通用的用法是设置reuse标志为AUTO_REUSE
5、reuse对tf.Variable定义的非共享变量是无用的
6、通过tf.Variable定义的变量任何时候都能被初始化或改变
'''
with tf.variable_scope(name_or_scope='scope_1'):
    w0=tf.get_variable(name='w0',shape=[1],dtype=tf.float32,initializer=tf.constant_initializer(1.0))
# tf.AUTO_REUSE
with tf.variable_scope(name_or_scope='scope_1',reuse=True):
    w1=tf.get_variable(name='w0',shape=[1],dtype=tf.float32,initializer=tf.constant_initializer(2.0))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(w0))
    print(sess.run(w1))


# 非共享变量tf.Variable
with tf.variable_scope(name_or_scope='scope_1'):
    #w0=tf.get_variable(name='w0',shape=[1],dtype=tf.float32,initializer=tf.constant_initializer(1.0))
    w0=tf.Variable(1.0,name='w0')
with tf.variable_scope(name_or_scope='scope_1',reuse=False):
    #w1=tf.get_variable(name='w1',shape=[1],dtype=tf.float32,initializer=tf.constant_initializer(2.0))
    w1=tf.Variable(2.0,name='w0')
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(w0))
    print(sess.run(w1))
