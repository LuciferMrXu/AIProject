import tensorflow as tf
import numpy as np

class LstmRNN(object):
    # 初始化
    def __init__(self,batch_size,poetrys_vector,vocab,model_choice,hidden_size,num_layers,embedding_size):
        self.batch_size=batch_size
        self.n_chunk=len(poetrys_vector)//batch_size # 迭代多少次
        self.poetrys_vector=poetrys_vector# 数据集
        self.vocab=vocab#字典
        self.model_choice=model_choice # 模型选择
        self.hidden_size=hidden_size # 神经元个数
        self.num_layers=num_layers  # 网络层数
        self.embedding_size=embedding_size # 词向量大小
    # 批数据生成
    def __get_batch(self):
        x_batches=[]
        y_batches=[]
        for i in range(self.n_chunk):
            start_index=i*self.batch_size
            end_index=(i+1)*self.batch_size
            batches=self.poetrys_vector[start_index:end_index]
            length=max(map(len,batches))
            # 使用self.vocab[' ']填充出一个shape为(self.batch_size,length)
            xdata=np.full((self.batch_size,length),self.vocab[' '],np.int32)
            for row in range(self.batch_size):
                xdata[row,:len(batches[row])]=batches[row]
            ydata=np.copy(xdata)
            ydata[:,:-1]=xdata[:,1:]
            x_batches.append(xdata)
            y_batches.append(ydata)
        return x_batches,y_batches

    def __network(self):
        self.input_data=tf.placeholder(tf.int32,[self.batch_size,None])
        self.output_targets=tf.placeholder(tf.int32,[self.batch_size,None])
        # rnn模型选择
        if self.model_choice=='gru':
            cell_func=tf.nn.rnn_cell.GRUCell
        elif self.model_choice=='lstm':
            cell_func=tf.nn.rnn_cell.BasicLSTMCell
        elif self.model_choice=='rnn':
            cell_func=tf.nn.rnn_cell.BasicRNNCell
        else:
            cell_func=tf.nn.rnn_cell.BasicRNNCell
        # 构建多层rnn
        cell=tf.nn.rnn_cell.MultiRNNCell([cell_func(self.hidden_size,state_is_tuple=True) for _ in range(self.num_layers)],state_is_tuple=True)
        # 初始化细胞状态
        initial_state=cell.zero_state(self.batch_size,tf.float32)

        # embedding
        embedding=tf.get_variable('embedding',[len(self.vocab),self.embedding_size])
        inputs=tf.nn.embedding_lookup(embedding,self.input_data)

        outputs,last_state=tf.nn.dynamic_rnn(cell,inputs,initial_state=initial_state)

        # softmax操作
        softmax_w=tf.get_variable('softmax_w',[self.hidden_size,len(self.vocab)+1])
        softmax_b=tf.get_variable('softmax_b',[len(self.vocab)+1])
        output=tf.reshape(outputs,[-1,self.hidden_size])
        logits=tf.nn.xw_plus_b(output,softmax_w,softmax_b)
        probs=tf.nn.softmax(logits)
        return logits,last_state,probs,cell,initial_state

    # 训练
    def train(self):
        logits,last_state,_,_,_=self.__network()
        targest=tf.reshape(self.output_targets,[-1])
        loss=tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],[targest],[tf.ones_like(targest,dtype=tf.float32)])
        cost=tf.reduce_mean(loss)
        # 梯度截断
        learning_rate=tf.Variable(0.0,trainable=False)
        # minimize =>(1)梯度计算 （2）梯度更新
        # 首先提取要训练的参数列表=》进行梯度计算gradients=》截断方法，截断梯度范围
        #global_step=tf.Variable(0,trainable=False)
        tvars=tf.trainable_variables()
        # clip_by_global_norm 梯度截断
        grads,_=tf.clip_by_global_norm(tf.gradients(cost,tvars),5)
        # 更新梯度
        optimier=tf.train.AdamOptimizer(learning_rate)
        # apply_gradients 应用梯度，梯度更新
        trian_op=optimier.apply_gradients(zip(grads,tvars))

        config=tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction=0.7
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver=tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state('model')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
            for epoch in range(50):
                sess.run(tf.assign(learning_rate,0.002*(0.97**epoch)))

                train_x,train_y=self.__get_batch()
                for batch_x,batch_y in zip(train_x,train_y):
                    trian_loss,_=sess.run([cost,trian_op],feed_dict={self.input_data:batch_x,self.output_targets:batch_y})
                    print(trian_loss)
                if epoch%10==0:
                    saver.save(sess,'model/poetry.model',global_step=epoch)


    # 预测
    def get_poetry(self):
        pass

    # 藏头诗
    def get_poetry_by_text(self,text):
        pass