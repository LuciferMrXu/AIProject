#_*_ coding:utf-8_*_
import tensorflow as tf

# input_x1=tf.constant(1)
# y=tf.add(input_x1,1)
# y_1=tf.add(y,2)
# with tf.Session() as sess:
#     for i in range(5):
#         y_out,y_1_out=sess.run([y,y_1])
#         print(y_out)
#         print(y_1_out)




input_x2=tf.placeholder(dtype=tf.float32,shape=None,name='input_x2')
input_x3=tf.placeholder(dtype=tf.float32,shape=None,name='input_x3')
y=tf.add(input_x2,input_x3)
y_1=tf.add(y,2)
# placeholder通过一次构图，计算链可以多次计算样本
with tf.Session() as sess:
    for i in range(5):
        y_out,y_1_out=sess.run([y,y_1],feed_dict={input_x2:i,input_x3:2})
        print(y_out)
        print(y_1_out)

