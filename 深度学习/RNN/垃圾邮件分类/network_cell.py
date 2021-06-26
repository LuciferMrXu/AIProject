import tensorflow as tf
class CNN:
    def __init__(self,sequence_length,num_class,vocab_size,embedding_size,filter_sizes,num_size,l2_reg_lambda):
        self.input_x=tf.placeholder(tf.int32,[None,sequence_length],name='input_x')
        self.input_y=tf.placeholder(tf.float32,[None,num_class],name='input_y')
        # dropout
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='dropout_keep_prob')

        l2_loss=tf.constant(0.0)

        # 词映射
        with tf.name_scope('embedding'):
            self.W=tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),dtype=tf.float32,name='W')
            self.embedded_chars=tf.nn.embedding_lookup(self.W,self.input_x)
            # 需要把三维的数据转换为四维
            self.embedded_chars_expanded=tf.expand_dims(self.embedded_chars,-1)

        #创建conv+pool
        pooled_outputs=[]
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv_maxpooling_%d'%i):
                filter_shape=[filter_size,embedding_size,1,num_size]

                W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
                b=tf.Variable(tf.constant(0.1,shape=[num_size]),name='b')

                conv=tf.nn.conv2d(self.embedded_chars_expanded,W,strides=[1,1,1,1],padding='VALID',name='CONV')
                relu=tf.nn.relu(tf.nn.bias_add(conv,b),name='Relu')
                pool=tf.nn.max_pool(relu,ksize=[1,sequence_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name='MaxPooling')
                pooled_outputs.append(pool)

        #合并所有汇集的特征
        num_filter_total=num_size*len(filter_sizes)
        self.h_pool=tf.concat(pooled_outputs,3)
        self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filter_total])

        # 全连接
        # （1）dropout
        with tf.name_scope('dropout'):
            self.h_drop=tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)

        # （2）fc
        with tf.name_scope('output'):
            W=tf.get_variable('W',shape=[num_filter_total,num_class],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.Variable(tf.constant(0.1,shape=[num_class],name='b'))

            # 正则化l2 计算损失
            l2_loss+=tf.nn.l2_loss(W)
            l2_loss+=tf.nn.l2_loss(b)

            # 评估
            self.scores=tf.nn.xw_plus_b(self.h_drop,W,b,name='scores')
            self.predictions=tf.argmax(self.scores,1,name='predictions')

        with tf.name_scope('loss'):
            losses=tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores,labels=self.input_y)
            self.loss=tf.reduce_mean(losses)+l2_reg_lambda*l2_loss

        with tf.name_scope('accurary'):
            correct_predictions=tf.equal(tf.argmax(self.input_y,1),self.predictions)
            self.accurary=tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuaray')

class RNN:
    def __init__(self,sequence_length,num_class,vocab_size,embedding_size,filter_sizes,num_size,l2_reg_lambda):
        self.input_x=tf.placeholder(tf.int32,[None,sequence_length],name='input_x')
        self.input_y=tf.placeholder(tf.float32,[None,num_class],name='input_y')
        # dropout
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='dropout_keep_prob')

        l2_loss=tf.constant(0.0)

        # 词映射
        with tf.name_scope('embedding'):
            self.W=tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),dtype=tf.float32,name='W')
            self.embedded_chars=tf.nn.embedding_lookup(self.W,self.input_x)
            # 需要把三维的数据转换为四维
            self.embedded_chars_expanded=tf.expand_dims(self.embedded_chars,-1)
        # 创建conv+pool
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv_maxpooling_%d' % i):
                filter_shape = [filter_size, embedding_size, 1, num_size]

                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_size]), name='b')

                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding='VALID',
                                    name='CONV')
                relu = tf.nn.relu(tf.nn.bias_add(conv, b), name='Relu')
                pool = tf.nn.max_pool(relu, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                      strides=[1, 1, 1, 1], padding='VALID', name='MaxPooling')
                pooled_outputs.append(pool)

        # 合并所有汇集的特征
        num_filter_total = num_size * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_reshape=tf.reshape(self.h_pool,[-1,num_size*3])
        self.word_char_features_dropout = tf.nn.dropout(self.h_pool_reshape, self.dropout_keep_prob,
                                                        name="word_char_features_dropout")

        #构建双向lstm
        rnn_cell_format = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_size, state_is_tuple=True, forget_bias=1.0),input_keep_prob=1.0,output_keep_prob=self.dropout_keep_prob)
        run_cell_backmat = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_size, state_is_tuple=True, forget_bias=1.0),output_keep_prob=self.dropout_keep_prob)
        #w*x+b
        # 运行网络
        # 输出结果，细胞状态
        # 生成双向rnn
        outputs, output_state_fw, output_state_bw = tf.nn.static_bidirectional_rnn(rnn_cell_format,run_cell_backmat, [self.word_char_features_dropout],dtype=tf.float32)
        grad_clip=5
        # 控制生成的结果的阀值（参考PPT 25页中的gradient clipping）
        self.biLstm_clip = tf.clip_by_value(outputs, -grad_clip, grad_clip)
        self.biLstm_dropout = tf.reshape(tf.nn.dropout(self.biLstm_clip, self.dropout_keep_prob),[-1,2*num_size])

        # （2）fc
        with tf.name_scope('output'):
            W=tf.get_variable('W',shape=[2*num_size,num_class],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.Variable(tf.constant(0.1,shape=[num_class],name='b'))

            # 正则化l2 计算损失
            l2_loss+=tf.nn.l2_loss(W)
            l2_loss+=tf.nn.l2_loss(b)

            # 评估
            print(self.biLstm_dropout)
            self.scores=tf.nn.xw_plus_b(self.biLstm_dropout,W,b,name='scores')

            self.predictions=tf.argmax(self.scores,1,name='predictions')

        with tf.name_scope('loss'):
            losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss=tf.reduce_mean(losses)+l2_reg_lambda*l2_loss

        with tf.name_scope('accurary'):
            correct_predictions=tf.equal(tf.argmax(self.input_y,1),self.predictions)
            self.accurary=tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuaray')
