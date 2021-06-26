# -*- coding:utf-8 -*-


import tensorflow as tf
a = tf.constant([1,2,3,4],dtype=tf.int32,shape=[2,2],name='a')
#你没有显示的定义图，表示你用了系统默认给你的图
print('常量a在默认的图上:{}'.format(a.graph is tf.get_default_graph()))
new_graph=tf.Graph()
with new_graph.as_default() as g1:
    new_a = tf.constant([[1,2],[1,2]], dtype=tf.int32,name='new_a')
    print('常量new_a在新的图上:{}'.format(new_a.graph is new_graph))
    print('常量new_a在默认的图上:{}'.format(new_a.graph is tf.get_default_graph()))
    print('常量a在默认的图上:{}'.format(a.graph is tf.get_default_graph()))
    b= tf.add(a,new_a)     #不同的图中数据op
