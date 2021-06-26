#_*_ coding:utf-8_*_
import tensorflow as tf

# 模型保存
v1 = tf.Variable(tf.constant(3.0), name='v1')
v2 = tf.Variable(tf.constant(4.0), name='v2')
result = v1 + v2

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(result)
    # 模型保存到model文件夹下，文件前缀为：model.ckpt
    saver.save(sess, './model/model.ckpt')



# 模型的提取
# 1、完整提取：需要完整恢复保存之前的数据格式
v1 = tf.Variable(tf.constant(1.0), name='v1')
v2 = tf.Variable(tf.constant(4.0), name='v2')
result = v1 + v2

saver = tf.train.Saver()
with tf.Session() as sess:
    # 会从对应的文件夹中加载变量、图等相关信息
    saver.restore(sess, './model/model.ckpt')
    print(sess.run([result]))



# 2、直接加载图，不需要定义变量了
saver = tf.train.import_meta_graph('./model/model.ckpt.meta')

with tf.Session() as sess:
    saver.restore(sess, './model/model.ckpt')
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))



# 3、模型的提取(给定映射关系)
a = tf.Variable(tf.constant(1.0), name='a')
b = tf.Variable(tf.constant(2.0), name='b')
result = a + b

saver = tf.train.Saver({"v1": a, "v2": b})
with tf.Session() as sess:
    # 会从对应的文件夹中加载变量、图等相关信息
    saver.restore(sess, './model/model.ckpt')
    print(sess.run([result]))
