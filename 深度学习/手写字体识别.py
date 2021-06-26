#_*_ coding:utf-8_*_
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


input_size=784
unit_size_1=256
unit_size_2=256
n_class=10
batch_size=50
epoch_num=100

mnist=input_data.read_data_sets('nn/data/', one_hot=True)
input_x=tf.placeholder(tf.float32,shape=[None,input_size],name='input_x')
#input_x=tf.placeholder(tf.float32,shape=[batch_size,input_size],name='input_x')
y=tf.placeholder(tf.float32,shape=[None,n_class],name='y')
#y=tf.placeholder(tf.float32,shape=[batch_size,n_class],name='y')

def layer(input_data,input_size,out_size,active_fun=None):
    w=tf.get_variable(name='w',shape=[input_size,out_size],dtype=tf.float32,
                      initializer=tf.random_normal_initializer())
    b=tf.get_variable(name='b',shape=[out_size],dtype=tf.float32,
                      initializer=tf.random_normal_initializer())
    output=tf.add(tf.matmul(input_data,w),b)

    if active_fun==None:
        out_put=output
    else:
        out_put=active_fun(output)
    return out_put


def build_net():
    with tf.variable_scope('layer1'):
        layer1=layer(input_x,input_size,unit_size_1,active_fun=tf.nn.sigmoid)
    with tf.variable_scope('layer2'):
        layer2 = layer(layer1,unit_size_1, unit_size_2, active_fun=tf.nn.sigmoid)
    with tf.variable_scope('logits'):
        logits=layer(layer2,unit_size_2,n_class)
    return logits


logits=build_net()

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=logits))
train=tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss)

acc=tf.equal(tf.argmax(logits,axis=1),tf.argmax(y,axis=1))
acc=tf.reduce_mean(tf.cast(acc, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # tf.global_variables_initializer/tf.local_variables_initializer
    epoch=0
    while epoch<epoch_num:
        sum_loss=0
        avg_loss=0
        batch_num=int(mnist.train.num_examples/batch_size)
        for i in range(batch_num):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            feeds={input_x:batch_xs,y:batch_ys}
            sess.run(train,feed_dict=feeds)
            sum_loss+=sess.run(loss,feeds)
        avg_loss=sum_loss/batch_num

        if (epoch + 1) % 4 == 0:
            print("批次: %03d 损失函数值: %.9f" % (epoch, avg_loss))
            feeds = {input_x: mnist.train.images, y: mnist.train.labels}
            train_acc = sess.run(acc, feed_dict=feeds)
            print("训练集准确率: %.3f" % train_acc)
            feeds = {input_x: mnist.test.images, y: mnist.test.labels}
            test_acc = sess.run(acc, feed_dict=feeds)
            print("测试准确率: %.3f" % test_acc)
        epoch += 1


