#encoding:utf-8
# 可视化
#将logs复制到系统盘

import tensorflow as tf

# with tf.variable_scope("foo"):
#     with tf.device("/cpu:0"):
#         x_init1 = tf.get_variable('init_x', [10], tf.float32, initializer=tf.random_normal_initializer())[0]
#         x = tf.Variable(initial_value=x_init1, name='x')
#         y = tf.placeholder(dtype=tf.float32, name='y')
#         z = x + y
# #
#     # update x
#     assign_op = tf.assign(x, x + 1)
#     with tf.control_dependencies([assign_op]):
#         with tf.device('/gpu:0'):
#             out = x * y
# #
# with tf.device('/cpu:0'):
#     with tf.variable_scope("bar"):
#         a = tf.constant(3.0) + 4.0
#     w = z * a
# #
# # 第一步：开始记录信息(需要展示的信息的输出)
# tf.summary.scalar('scalar_init_x', x_init1)
# tf.summary.scalar(name='scalar_x', tensor=x)
# tf.summary.scalar('scalar_y', y)
# tf.summary.scalar('scalar_z', z)
# tf.summary.scalar('scala_w', w)
# tf.summary.scalar('scala_out', out)
# #
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     # 第二步：merge all summary
#     merged_summary = tf.summary.merge_all()
#     # 第三步：得到输出到文件的对象
#     writer = tf.summary.FileWriter('./result', sess.graph)
# #
#     # 初始化
#     sess.run(tf.global_variables_initializer())
#     # print
#     for i in range(1, 5):
#         summary, r_out, r_x, r_w = sess.run([merged_summary, out, x, w], feed_dict={y: i})
#         writer.add_summary(summary, i)
#         print("{},{},{}".format(r_out, r_x, r_w))
# #
#     # 关闭操作
#     writer.close()

# 简化版
input_x = tf.placeholder(dtype=tf.float32,shape=None,name='input_x')
w = tf.Variable(1.0,name='w')
y = tf.add(input_x,w)

#1.开始在加载scalar
'''
真正的项目中：tf.summary.scalar(name='loss',tensor='loss')
'''
tf.summary.scalar(name='input_x',tensor=input_x)
tf.summary.scalar('w',w)
tf.summary.scalar('y',y)

with tf.Session() as sess:
    #2. 合并所有的scalar
    summ_merg = tf.summary.merge_all()
    #3. 定义summary wirter
    summ_writer = tf.summary.FileWriter('logs',sess.graph) # logs是当前目录下的文件夹
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        #4.计算merge
        merg_out = sess.run(summ_merg,feed_dict={input_x:i})
        #5.写merge
        summ_writer.add_summary(merg_out,i)



