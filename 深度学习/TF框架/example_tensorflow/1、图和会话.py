# -- encoding:utf-8 --
"""
Create by ibf on 2018/5/5
"""

import tensorflow as tf

# 1. 定义常量矩阵a和矩阵b
# name属性只是给定这个操作一个名称而已
a = tf.constant([[1, 2], [3, 4]], dtype=tf.int32, name='a')
print(type(a))
b = tf.constant([5, 6, 7, 8], dtype=tf.int32, shape=[2, 2], name='b')

# 2. 以a和b作为输入，进行矩阵的乘法操作
c = tf.matmul(a, b, name='matmul')
print(type(c))

# 3. 以a和c作为输入，进行矩阵的相加操作
g = tf.add(a, c, name='add')
print(type(g))
print(g)

# 4. 添加减法
h = tf.subtract(b, a, name='b-a')
l = tf.matmul(h, c)
r = tf.add(g, l)



# # 使用新的构建的图
# graph = tf.Graph()
# with graph.as_default():
#     # 此时在这个代码块中，使用的就是新的定义的图graph(相当于把默认图换成了graph)
#     d = tf.constant(5.0, name='d')
#     print("变量d是否在新图graph中:{}".format(d.graph is graph))
#
#
# with tf.Graph().as_default() as g2:
#     e = tf.constant(6.0)
#     print("变量e是否在新图g2中：{}".format(e.graph is g2))
#
#
# print("变量d是否在原来的默认图中:{}".format(d.graph is tf.get_default_graph()))


'''
f = tf.add(d, e)
这段代码是错误的用法，记住：不能使用两个图中的变量进行操作，
只能对同一个图中的变量对象（张量）进行操作(op)
'''


# 会话构建&启动(默认情况下（不给定Session的graph参数的情况下），创建的Session属于默认的图)
sess = tf.Session()
print(sess)


'''
调用sess的run方法来执行矩阵的乘法，得到c的结果值（所以将c作为参数传递进去）
不需要考虑图中间的运算，在运行的时候只需要关注最终结果对应的对象以及所需要的输入数据值
只需要传递进去所需要得到的结果对象，会自动的根据图中的依赖关系触发所有相关的OP操作的执行
如果op之间没有依赖关系，tensorflow底层会并行的执行op(有资源) --> 自动进行
如果传递的fetches是一个列表，那么返回值是一个list集合
fetches：表示获取对应的op操作的结果值
'''


result = sess.run(fetches=[r, c])
print("type:{}, value:\n{}".format(type(result), result))

# 会话关闭
sess.close()


'''
当一个会话关闭后，不能再使用了，所以下面两行代码错误
result2 = sess.run(c)
print(result2)
'''


# 使用with语句块，会在with语句块执行完成后，自动的关闭session
# sess2=tf.Session()
# with sess2.as_default():

gpu_options=tf.GPUOptions()
gpu_options.per_process_gpu_memory_fraction=0.75
gpu_options.allow_growth=True

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=True,allow_soft_placement=True)) as sess2:
    print(sess2)
    # 获取张量c的结果： 通过Session的run方法获取
    print("sess2 run:{}".format(sess2.run(c)))
    # 获取张量r的结果：通过张量对象的eval方法获取，和Session的run方法一致
    print("c eval:{}".format(r.eval()))

# 交互式会话构建
sess3 = tf.InteractiveSession()
print(r.eval())

