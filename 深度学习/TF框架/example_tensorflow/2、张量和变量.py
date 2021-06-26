#_*_ coding:utf-8_*_
import tensorflow as tf
# 1. 定义一个变量，必须给定初始值(图的构建，没有运行)
a = tf.Variable(initial_value=3.0, dtype=tf.float32)

# 2. 定义一个张量(这里是常量constant)
b = tf.constant(value=2.0, dtype=tf.float32)
c = tf.add(a, b)

# 3. 进行初始化操作（推荐：使用全局所有变量初始化API）
# 相当于在图中加入一个初始化全局变量的操作
init_op = tf.global_variables_initializer()
print(type(init_op))

# 4. 图的运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 运行init op进行变量初始化，一定要放到所有运行操作之前
    sess.run(init_op)
    # init_op.run() # 这行代码也是初始化运行操作，但是要求明确给定当前代码块对应的默认session(tf.get_default_session())是哪个，底层使用默认session来运行
    # 获取操作的结果
    print("result:{}".format(sess.run(c)))
    print("result:{}".format(c.eval()))

'''
    构建变量间的依赖
'''
# 1. 定义变量，常量
w1 = tf.Variable(tf.random_normal(shape=[10], stddev=0.5, seed=28, dtype=tf.float32), name='w1') # stddev是正态分布的方差
a = tf.constant(value=2.0, dtype=tf.float32)
w2 = tf.Variable(w1.initialized_value() * a, name='w2')   # 最好用w1初始化后的值构建w2的依赖

# 3. 进行初始化操作（推荐：使用全局所有变量初始化API）
# 相当于在图中加入一个初始化全局变量的操作
init_op = tf.global_variables_initializer()
print(type(init_op))

# 3. 图的运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 运行init op进行变量初始化，一定要放到所有运行操作之前
    sess.run(init_op)
    # init_op.run() # 这行代码也是初始化运行操作，但是要求明确给定当前代码块对应的默认session(tf.get_default_session())是哪个，底层使用默认session来运行
    # 获取操作的结果
    print("result:{}".format(sess.run(w1)))
    print("result:{}".format(w2.eval()))

'''
    fetch 和 feed
    fetch是获取节点的输出值，用session.run(fetches=节点)或者  节点.eval()获取 
    feed是一个填充机制，用来填充占位符，调用结束后，填充的数据消失
'''
# 构建一个矩阵的乘法，但是矩阵在运行的时候给定
m1 = tf.placeholder(dtype=tf.float32, shape=[2, 3], name='placeholder_1')
m2 = tf.placeholder(dtype=tf.float32, shape=[3, 2], name='placeholder_2')
m3 = tf.matmul(m1, m2)

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    print("result:\n{}".format(
        sess.run(fetches=m3, feed_dict={m1: [[1, 2, 3], [4, 5, 6]], m2: [[9, 8], [7, 6], [5, 4]]})))
    print("result:\n{}".format(m3.eval(feed_dict={m1: [[1, 2, 3], [4, 5, 6]], m2: [[9, 8], [7, 6], [5, 4]]})))
