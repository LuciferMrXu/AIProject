import data_helper
import numpy as np
from tensorflow.contrib import learn
import network_cell
import tensorflow as tf
import time
import os # 系统文件和目录操作

print('Loading data ...')
positive_data_file='data/rt-polaritydata/rt-polarity.pos'
negative_data_file='data/rt-polaritydata/rt-polarity.neg'
X,Y=data_helper.load_data_and_label(positive_data_file,negative_data_file)

# 传入神经网络的数据为每一行，那么我们就需要进行保证数据矩阵的大小一致性
# 数据填充
# 找到邮件长度最长的那个
max_document_length=max([len(x.split(" ")) for x in X])
# 传入最大的长度，默认填充0
vocab_processor=learn.preprocessing.VocabularyProcessor(max_document_length)
X=np.array(list(vocab_processor.fit_transform(X)))

# 打乱数据，并生成验证集
np.random.seed(10)
shuffle_index=np.random.permutation(np.arange(len(Y)))
x_shuffled=X[shuffle_index]
y_shuffled=Y[shuffle_index]

dev_sample_index=-1*int(0.1*float(len(Y)))# 比例
X_train,X_test=x_shuffled[:dev_sample_index],x_shuffled[dev_sample_index:]
Y_train,Y_test=y_shuffled[:dev_sample_index],y_shuffled[dev_sample_index:]

print('Vocabulary Size:%d'%len(vocab_processor.vocabulary_))
print("Train/Test split:%d/%d"%(len(Y_train),len(Y_test)))

# tf,FLAGS

# 构建网络（network）
# 序列长度，按照时间顺序展开，x数量
sequence_length=X_train.shape[1]
# 分成多少类
num_class=Y_train.shape[1]
# 词典大小
vocab_size=len(vocab_processor.vocabulary_)
# 词向量隐射的维度大小
# 每个单词转换为向量维度是128
embedding_size=128
#窗口大小
filter_sizes=[3,4,5]
#神经元个数
num_size=100
# L2惩罚项
l2_reg_lambda=0.0

sess=tf.Session()
#network=network.CNN(sequence_length,num_class,vocab_size,embedding_size,filter_sizes,num_size,l2_reg_lambda)
network=network_cell.RNN(sequence_length, num_class, vocab_size, embedding_size, filter_sizes, num_size, l2_reg_lambda)
# 优化器
# trainable 变量是否参与训练
global_step=tf.Variable(0,name='global_step',trainable=False)
optimizer=tf.train.AdamOptimizer(1e-5)
grads_and_vars=optimizer.compute_gradients(network.loss)
# 使用优化求解的方式计算network
train_op=optimizer.apply_gradients(grads_and_vars,global_step=global_step)

# 开启日志管理事件
grad_summaries=[]
# 跟踪稀疏和梯度值
for g,v in grads_and_vars:
    if g is not None:
        # 历史变化
        grad_hist_summary=tf.summary.histogram('{}/grad/hist'.format(v.name),g)
        # 梯度过程
        sparsity_summary=tf.summary.scalar('{}/grad/sparsity'.format(v.name),tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
grads_summaries_merged=tf.summary.merge(grad_summaries)
# 创建模型和摘要
timestamp=str(int(time.time()))
timestamp = '1521371378'
out_dir=os.path.abspath(os.path.join(os.path.curdir,'runs',timestamp))

# 记录损失和准确率
loss_summary=tf.summary.scalar('loss',network.loss)
acc_summary=tf.summary.scalar('accurary',network.accurary)
train_summary_op=tf.summary.merge([loss_summary,acc_summary,grads_summaries_merged])
train_summary_dir=os.path.join(out_dir,'summary','train')

# 记录
train_summary_writer=tf.summary.FileWriter(train_summary_dir,sess.graph)

dev_summary_op=tf.summary.merge([loss_summary,acc_summary])
dev_summary_dir=os.path.join(out_dir,'summary','test')
dev_summary_writer=tf.summary.FileWriter(dev_summary_dir,sess.graph)

# 模型持久化
saver=tf.train.Saver(tf.global_variables(),max_to_keep=5)

# 记录字典
vocab_processor.save(os.path.join(out_dir,'vocab'))

sess.run(tf.global_variables_initializer())

# train_step
def train_step(X_batch,Y_batch):
    global network
    feed_dict={
        network.input_x:X_batch,
        network.input_y:Y_batch,
        network.dropout_keep_prob:0.75
    }
    _,step,summaries,loss,acc=sess.run([train_op, global_step, train_summary_op, network.loss, network.accurary], feed_dict=feed_dict)
    train_summary_writer.add_summary(summaries)
    print("step {},loss:{},acc:{}".format(step,loss,acc))

def dev_step(X_batch,Y_batch):
    feed_dict={
        network.input_x:X_batch,
        network.input_y:Y_batch,
        network.dropout_keep_prob:1.0
    }
    step,summaries,loss,acc=sess.run([global_step, dev_summary_op, network.loss, network.accurary], feed_dict=feed_dict)
    dev_summary_writer.add_summary(summaries)
    print("step {},loss:{},acc:{}".format(step,loss,acc))

batches=data_helper.batch_iter(list(zip(X_train,Y_train)),64,num_size)
for batch in batches:
    x_batch,y_batch=zip(*batch)
    train_step(x_batch,y_batch)
    current_step=tf.train.global_step(sess,global_step)
    if current_step%10==0:
        dev_step(x_batch,y_batch)
    if current_step%100==0:
        saver.save(sess,os.path.join(out_dir,'model.ckpt','email_network'),global_step=current_step)