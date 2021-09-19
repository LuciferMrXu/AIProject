#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
    使用TFIDF提取文本特征
    使用keras神经网络
    Score = ?
"""
import pandas as pd
import pickle

with open('./df_train.pkl', 'rb') as file:
    df_train = pickle.load(file)
with open('./df_test.pkl', 'rb') as file:
    df_test = pickle.load(file)
    
with open('./df_apis.pkl', 'rb') as file:
    df_apis = pickle.load(file)
df_apis


# In[2]:


# 13887
df_train_apis = df_apis[df_apis.index <=13886]
# 12955
df_test_apis = df_apis[df_apis.index > 13886]
df_train_apis


# In[23]:


df_train.to_pickle('./df_train2.pkl')
df_test.to_pickle('./df_test2.pkl')


# In[3]:


df_train.index = range(len(df_train))
df_test_apis.index = range(len(df_test))
df_test_apis


# In[4]:


df_train = df_train.merge(df_train_apis, left_index=True, right_index=True)
df_train


# In[5]:


# merge 将不同的列进行合并 => 列增多了
# concat 将不同的行进行合并 => 行增多了
df_test.index = range(len(df_test))
df_test = df_test.merge(df_test_apis, left_index=True, right_index=True)
df_test


# In[6]:


df_train[['index']]
# 去掉没用的列
df_train.drop(['api', 'tid', 'index'], axis=1, inplace=True)
df_test.drop(['api', 'tid', 'index'], axis=1, inplace=True)
df_train.columns


# In[7]:


# 特征列
df_train_x = df_train.drop(['file_id', 'label'], axis=1)
# label
df_train_y = df_train['label']
df_test_x = df_test.drop(['file_id'], axis=1)


# In[8]:


for col in df_test_x.columns:
    num = df_test_x[col].isnull().sum()
    if num > 0:
        print(col, num)
# 填充缺失值
df_train_x.fillna(0, inplace=True)
df_test_x.fillna(0, inplace=True)


# In[9]:


#df_train_x.shape[1]
# 对于神经网络，需要做特征归一化处理
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_train_x = scaler.fit_transform(df_train_x)
df_test_x = scaler.transform(df_test_x)


# In[16]:


from tensorflow.keras.utils import to_categorical
from tensorflow import keras
# 将y转换为 one-hot编码
to_categorical(df_train_y)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers

# 定义网络模型
model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=[df_train_x.shape[1]], kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    # 转换为多分类的概率
    keras.layers.Dense(8, activation='softmax')
])
# 定义优化器
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# 模型训练
model.fit(df_train_x, to_categorical(df_train_y), batch_size=1024, epochs=10, verbose=1)


# In[17]:


# 模型训练
model.fit(df_train_x, to_categorical(df_train_y), batch_size=1024, epochs=20, verbose=1)


# In[18]:


# 模型预测
result = model.predict_proba(df_test_x)
result


# In[19]:


#model.fit(df_train_x, to_categorical(df_train_y), batch_size=1024, epochs=50, verbose=1)
result = pd.DataFrame(result, columns=['prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7'])
result


# In[20]:


result['file_id'] = df_test['file_id'].values
result


# In[21]:


#df_test['file_id'].values
columns = ['file_id', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7']
result.to_csv('baseline_tfidf_nn2.csv', index=False, columns=columns)

