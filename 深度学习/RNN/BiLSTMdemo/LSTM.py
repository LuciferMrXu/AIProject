#encoding=utf8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data', one_hot=True)

# 参数设置
BATCH_SIZE = 100        # BATCH的大小，相当于一次处理50个image
TIME_STEP = 28          # 一个LSTM中，输入序列的长度，image有28行
INPUT_SIZE = 28         # x_i 的向量长度，image有28列
LR = 0.01               # 学习率
NUM_UNITS = 100         # 多少个LTSM单元
ITERATIONS=8000         # 迭代次数
N_CLASSES=10            # 输出大小，0-9十个数字的概率

# 定义 placeholders 以便接收x,y
train_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE])       # 维度是[BATCH_SIZE，TIME_STEP * INPUT_SIZE]
image = tf.reshape(train_x, [-1, TIME_STEP, INPUT_SIZE])                   # 输入的是二维数据，将其还原为三维，维度是[BATCH_SIZE, TIME_STEP, INPUT_SIZE]
train_y = tf.placeholder(tf.int32, [None, N_CLASSES])                     

# 定义RNN（LSTM）结构
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=NUM_UNITS) 
outputs,final_state = tf.nn.dynamic_rnn(
    cell=rnn_cell,              # 选择传入的cell
    inputs=image,               # 传入的数据
    initial_state=None,         # 初始状态
    dtype=tf.float32,           # 数据类型
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)，这里根据image结构选择False
)
output = tf.layers.dense(inputs=outputs[:, -1, :], units=N_CLASSES)   

loss = tf.losses.softmax_cross_entropy(onehot_labels=train_y, logits=output)      # 计算loss
train_op = tf.train.AdamOptimizer(LR).minimize(loss)      #选择优化方法

correct_prediction = tf.equal(tf.argmax(train_y, axis=1),tf.argmax(output, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))  #计算正确率

sess = tf.Session()
sess.run(tf.global_variables_initializer())     # 初始化计算图中的变量

for step in range(ITERATIONS):    # 开始训练
    x, y = mnist.train.next_batch(BATCH_SIZE)   
    test_x, test_y = mnist.test.next_batch(5000)
    _, loss_ = sess.run([train_op, loss], {train_x: x, train_y: y})
    if step % 50 == 0:      # test（validation）
        accuracy_ = sess.run(accuracy, {train_x: test_x, train_y: test_y})
        print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)