#encoding:utf-8
import tensorflow as tf

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    with tf.variable_scope('scope_1', initializer=tf.constant_initializer(4.0)) as scope_1:
        a_1 = tf.get_variable("a_1", [1])
        a_2 = tf.get_variable("a_2", [1], initializer=tf.constant_initializer(3.0))
        with tf.variable_scope('scope_2'):
            scope_2_a1 = tf.get_variable("scope_2_a1", [1])

            with tf.variable_scope(scope_1):  #注意：对get_variable和Variable不同效果
                scope_1_a3 = tf.get_variable('scope_1_a3', [1])  #直接使用最上层域名对象scope_1的域名作为域名
                add_1 = a_1 + a_2 + scope_2_a1 + scope_1_a3      #多层嵌套scope_1/scope_2/scope_1；

    with tf.variable_scope('abc'):
        a = tf.get_variable('a', [1], initializer=tf.constant_initializer(5.0))
        b = a + add_1

    sess.run(tf.global_variables_initializer())
    print("{},{}".format(a_1.name, a_1.eval()))
    print("{},{}".format(a_2.name, a_2.eval()))
    print("{},{}".format(scope_2_a1.name, scope_2_a1.eval()))
    print("{},{}".format(scope_1_a3.name, scope_1_a3.eval()))
    print("{},{}".format(add_1.name, add_1.eval()))
    print("{},{}".format(a.name, a.eval()))
    print("{},{}".format(b.name, b.eval()))


#简化版
'''
1、变量作用域可以嵌套
2、变量作用域对tf.Variable和tf.get_variable定义的变量都产生作用
3、通过传入别名定义变量作用域，只对get_variable共享变量有作用域层次提升效果
'''
with tf.variable_scope('scope_1') as scope_x:
    # w1=tf.get_variable(name='w1',dtype=tf.float32,shape=[1],initializer=tf.constant_initializer(0,1))
    w1=tf.Variable(1.0,name='w1')
    with tf.variable_scope('scope_2'):
        # w2=tf.get_variable(name='w2',dtype=tf.float32,shape=[1],initializer=tf.constant_initializer(0,2))
        w2=tf.Variable(2.0,name='w2')
        with tf.variable_scope(scope_x):
            w3_get = tf.get_variable(name='w3', dtype=tf.float32, shape=[1], initializer=tf.constant_initializer(0,3))
            w3=tf.Variable(3.0,name='w3')
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print("{},{}".format(w1.name, w1.eval()))
    print("{},{}".format(w2.name, w2.eval()))
    print("{},{}".format(w3_get.name, w3_get.eval()))
    print("{},{}".format(w3.name, w3.eval()))