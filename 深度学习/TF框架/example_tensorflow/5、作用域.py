#_*_ coding:utf-8_*_
import tensorflow as tf


'''
    当模型用到多个变量时
'''
# 常规写法
def my_func(x):
    w1 = tf.Variable(tf.random_normal([1]))[0]
    b1 = tf.Variable(tf.random_normal([1]))[0]
    r1 = w1 * x + b1

    w2 = tf.Variable(tf.random_normal([1]))[0]
    b2 = tf.Variable(tf.random_normal([1]))[0]
    r2 = w2 * r1 + b2

    return r1, w1, b1, r2, w2, b2


# 下面两行代码还是属于图的构建
x = tf.constant(3, dtype=tf.float32)
r = my_func(x)

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 初始化
    tf.global_variables_initializer().run()
    # 执行结果
    print(sess.run(r))



# 构建变量作用域的写法
def my_func(x):
    # initializer：初始化器
    # w = tf.Variable(tf.random_normal([1]), name='w')[0]
    # b = tf.Variable(tf.random_normal([1]), name='b')[0]
    w = tf.get_variable(name='w', shape=[1], initializer=tf.random_normal_initializer())[0]
    b = tf.get_variable(name='b', shape=[1], initializer=tf.random_normal_initializer())[0]
    r = w * x + b

    return r, w, b


def func(x):
    with tf.variable_scope('op1', reuse=tf.AUTO_REUSE):
        r1 = my_func(x)
    with tf.variable_scope('op2', reuse=tf.AUTO_REUSE):
        r2 = my_func(r1[0])
    return r1, r2


# 下面两行代码还是属于图的构建
x1 = tf.constant(3, dtype=tf.float32, name='x1')
x2 = tf.constant(4, dtype=tf.float32, name='x2')
with tf.variable_scope('func1'):
    r1 = func(x1)
with tf.variable_scope('func2'):
    r2 = func(x2)

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 初始化
    tf.global_variables_initializer().run()
    # 执行结果
    print(sess.run([r1, r2]))





'''
变量作用域的嵌套
'''
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    with tf.variable_scope('foo', initializer=tf.constant_initializer(4.0)) as foo:
        v = tf.get_variable("v", [1])
        w = tf.get_variable("w", [1], initializer=tf.constant_initializer(3.0))
        with tf.variable_scope('bar'):
            l = tf.get_variable("l", [1])

            with tf.variable_scope(foo):
                h = tf.get_variable('h', [1])
                g = v + w + l + h

    with tf.variable_scope('abc'):
        a = tf.get_variable('a', [1], initializer=tf.constant_initializer(5.0))
        b = a + g

    sess.run(tf.global_variables_initializer())
    print("{},{}".format(v.name, v.eval()))
    print("{},{}".format(w.name, w.eval()))
    print("{},{}".format(l.name, l.eval()))
    print("{},{}".format(h.name, h.eval()))
    print("{},{}".format(g.name, g.eval()))
    print("{},{}".format(a.name, a.eval()))
    print("{},{}".format(b.name, b.eval()))
