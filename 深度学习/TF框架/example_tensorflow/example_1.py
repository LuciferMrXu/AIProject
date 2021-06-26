#coding:utf-8
import tensorflow as tf

# 需求一
# 1. 定义一个变量
x_1 = tf.Variable(1, dtype=tf.int32, name='x_1')

# 2. 变量的更新,其中 ref是更新前的值x_1，value是更新后的值从新赋值给x_1,并返回给assign_update;类似j=++i
assign_update = tf.assign(ref=x_1,value=x_1 + 1)

# 3. 运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 变量初始化
    sess.run(tf.initialize_all_variables())
    # 模拟迭代更新累加器
    for i in range(5):
        # 执行更新操作
        print(sess.run(assign_update))
        #r_x = sess.run(x_1)
        #print(r_x)


# 需求二
# 1. 定义一个不定形状的变量
x = tf.Variable(
    initial_value=[],  # 给定一个空值
    dtype=tf.float32,
    trainable=True,       #训练模型的时候为真进行BP，测试模型的时候为False
    validate_shape=False  # 设置为True，表示在变量更新的时候，进行shape的检查，默认为True
)
#
# 2. 变量更改
concat_1 = tf.concat([x, [0.0]], axis=0)
assign_update = tf.assign(ref=x, value=concat_1,validate_shape=False)

#
# 3. 运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 变量初始化
    sess.run(tf.initialize_all_variables())
#
    # 模拟迭代更新累加器
    for i in range(5):
        # 执行更新操作
        print(sess.run(assign_update))
   



# 需求三
# 1. 定义一个变量
sum = tf.Variable(1, dtype=tf.int32)
# 2. 定义一个占位符
i = tf.placeholder(dtype=tf.int32)

# 3. 更新操作
tmp_sum = sum * i
# tmp_sum = tf.multiply(sum, i)
assign_update = tf.assign(sum, tmp_sum)

with tf.control_dependencies([assign_update]):
    # 如果需要执行这个代码块中的内容，必须先执行control_dependencies中给定的操作/tensor
    #sum = tf.Print(sum, data=[sum, sum.read_value()], message='sum:')
    #sum必须有新操作tf相关
    sum1 = tf.add(sum,1)


# 4. 运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 变量初始化
    sess.run(tf.initialize_all_variables())

    # 模拟迭代更新累加器
    for j in range(1, 6):
        # 执行更新操作
        # sess.run(assign_op, feed_dict={i: j})
        # 1.通过control_dependencies可以指定依赖关系，这样的话，就不用管内部的更新操作了
        r1 = sess.run(sum1,feed_dict={i:j})
        print(r1)
        # 2.直接使用，无新op不需要有依赖
        #r = sess.run(assign_update, feed_dict={i:j})

    print("5!={}".format(r1))
   

#简化版
# assign产生依赖
'''
1、w+input_x 赋值给了assign_update，同时也对w重新赋值（更新一次）
2、要执行代码块，必须先执行依赖的操作控制assign_update
'''
input_x=tf.placeholder(dtype=tf.float32,shape=None,name='input_x')
w=tf.Variable(1.0,name='w')
assign_update=tf.assign(w,w+input_x)
with tf.control_dependencies([assign_update]):
    y=tf.add(w,0.0)   # 代码块
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(5):
        # 第一种
        # print(sess.run(assign_update,feed_dict={input_x:i}))
        # 第二种
        print(sess.run(y,feed_dict={input_x:i}))
















