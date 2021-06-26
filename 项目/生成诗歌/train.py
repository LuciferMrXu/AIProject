from load_data import *
from lstm_models import *
# 读取数据
poertrys=data_load()
# 构建字典
vocab,de_vocab=get_vocab(poertrys)
# 转换文字到字典下标的形式
poetrys_vector=word2vec(vocab,poertrys)

# 初始化模型，执行训练
rnn_net=LstmRNN(batch_size=64,poetrys_vector=poetrys_vector,vocab=vocab,model_choice='lstm',hidden_size=128,num_layers=2,embedding_size=128)
rnn_net.train()

