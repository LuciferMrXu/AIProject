#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle

# 数据加载
"""
with open('./df_train.pkl', 'rb') as file:
    df_train = pickle.load(file)
with open('./df_test.pkl', 'rb') as file:
    df_test = pickle.load(file)
df_train
"""
df_train = pd.read_csv('./security_train.csv')
df_test = pd.read_csv('./security_test.csv')
df_train


# In[2]:


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical, pad_sequences
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers

df_train_y = df_train['label']
max_len = 6000
# 将y转换为 one-hot编码
labels = to_categorical(df_train_y)


# In[3]:


#df_train[['api']]
# 得到字典
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train['api'].tolist())
tokenizer.fit_on_texts(df_test['api'].tolist())


# In[4]:


# 查看vocab大小
vocab = tokenizer.word_index
vocab


# In[5]:


# 将句子，用字典ID进行变换
x_train_word_ids = tokenizer.texts_to_sequences(df_train['api'].tolist())
x_test_word_ids = tokenizer.texts_to_sequences(df_test['api'].tolist())


# In[9]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
# 填充到固定长度 max_len （长的截取，短的补）
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=max_len)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=max_len)


# In[ ]:


import pickle
with open('./vocab.pkl', 'wb') as file:
    pickle.dump(vocab, file)
with open('./x_train_padded_seqs.pkl', 'wb') as file:
    pickle.dump(x_train_padded_seqs, file)
with open('./x_test_padded_seqs.pkl', 'wb') as file:
    pickle.dump(x_test_padded_seqs, file)


# In[ ]:


len(vocab)


# In[ ]:


def fasttext():
    # 搭建模型
    model = keras.Sequential([
        keras.layers.Embedding(len(vocab), 256, input_length=max_len),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(8, activation='softmax')
    ])
    return model

from sklearn.model_selection import train_test_split
X, y = x_train_padded_seqs, labels
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2021)
model = fastext()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
history = model.fit(X, y, batch_size=1024, epochs=1)


# In[ ]:


result = mode.predict(x_test_padded_seqs)


# In[ ]:


result = pd.DataFrame(result, columns=['prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7'])
result
result['file_id'] = df_test['file_id'].values
result
columns = ['file_id', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7']
result.to_csv('baseline_fasttext.csv', index=False, columns=columns)

