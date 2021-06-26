# -*- coding:utf-8 -*-

import tensorflow as tf

a_1 = tf.constant([[1,2],[1,3]],dtype=tf.int32,name='a_1')
a_2 = tf.constant([[2,1],[3,1]],dtype=tf.int32,name='a_2')
a_add = tf.add(a_1,a_2)
a_matmul = tf.matmul(a_1,a_2)
a_mul = tf.multiply(a_1,a_2)
a_sub = tf.subtract(a_add,a_mul)
b_1 = tf.placeholder(dtype=tf.int32,shape=[None,2])
#1.定义占位符x 2. 用占位符构建计算链条，链条最后输出是r 3.想知道计算链条结果r，run(r,
#同时通过feed_dict={x:?}将数据传入) 
# session的开启与关闭方法一：

# sess=tf.Session(target='',graph='',config='')
#op
#sess.close()

b_a = tf.matmul(a_1,b_1)
# session的开启与关闭方法二,使用缺省的图：
'''
1、Session一定要自己显式的调用
2、Session一般只能有一个，初始化module时用一次即可
'''
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # 在图中执行操作
    #1. 可以取出任何一个位置的操作
    #2. 非相关操作可以并行进行
    #3. run有回溯能力，从最后结果，根据图中的op依赖关系，回溯到初始的feed(placeholder)定义的量或tf.constant定义的常量
    #4. fetches可以从任何节点获取结果值，一般省略不写，放在第一位的参数即为fetches;它可以是列表，列表中位置可以互换；函数输出对应的是fetches参数计算的结果；
    #5. feed_dict可以传入新的数据，作为当前运算 
    print("a_sub is：")
    print(sess.run(a_sub))
    print("a_add is: ")
    print(sess.run(a_add))
    print("matMul is :")
    print(sess.run(a_matmul))
    print("mul is :")
    print(sess.run(a_mul))
    print("fetch list is ：")
    print(sess.run(fetches=[a_add,a_sub]))
    #可以加入循环
    print("placeHolder is: ")
    print(sess.run(b_a, feed_dict={b_1:[[1,2],[1,2]]}))
    #sess.run(fetches)
    #在内部op的危害
    d = tf.add(a_1,a_2)
    print(sess.run(d))
  

#简化版
