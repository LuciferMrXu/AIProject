#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# 数据加载
def get_data(file_name):
    result = []
    chunk_index = 0
    for df in pd.read_csv(open(file_name, 'r'), chunksize = 1000000):
        result.append(df)
        print('chunk', chunk_index)
        chunk_index += 1
    result = pd.concat(result, ignore_index=True, axis=0)
    return result

# 获取全量数据
train = get_data('./security_train.csv')
train


# In[2]:


# 获取全量数据
test = get_data('./security_test.csv')
test


# In[3]:


# 13887个file_id
train['file_id'].value_counts()


# In[4]:


import os
import psutil
mem = psutil.virtual_memory()
print('总内存：',mem.total/1024/1024)
print('已使用内存：', mem.used/1024/1024)
print('空闲内存：', mem.free/1024/1024)
print('使用占比：',mem.percent)
print('当前线程PID：', os.getpid())


# In[5]:


# 文件以python对象格式进行保存
import pickle
with open('./train.pkl', 'wb') as f:
    pickle.dump(train, f)

with open('./test.pkl', 'wb') as f:
    pickle.dump(test, f)


# In[6]:


import pandas as pd
# 对api字段进行LabelEncoder
#train['api'].describe()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# 将训练集 和 测试集进行合并
df_all = pd.concat([train, test])
df_all['api'] = le.fit_transform(df_all['api'])
df_all[['api']]


# In[7]:


# 提取train['api']
train['api'] = df_all[df_all['label'].notnull()]['api']
test['api'] = df_all[df_all['label'].isnull()]['api']
train


# In[8]:


# 针对不用的变量，进行内存释放
import gc
del df_all
gc.collect()


# In[9]:


# 构造新的特征（基于file_id的聚合统计）
def get_features(df):
    # 按照file_id分组，提取统计特征
    df_file = df.groupby('file_id')
    # df1为最终的结果
    if 'label' in df.columns: 
        # 训练集
        df1 = df.drop_duplicates(subset=['file_id', 'label'], keep='first')
    else: 
        # 测试集
        df1 = df.drop_duplicates(subset=['file_id'], keep='first')
        df1 = df1.sort_values('file_id')
    # 提取多个特征的 统计特征 api, tid, index
    features = ['api', 'tid', 'index']
    for f in features:
        # 针对file_id 构造不同特征， 一个file_id 只有一行数据
        df1[f+'_count'] = df_file[f].count().values
        df1[f+'_nunique'] = df_file[f].nunique().values
        df1[f+'_min'] = df_file[f].min().values
        df1[f+'_max'] = df_file[f].max().values
        df1[f+'_mean'] = df_file[f].mean().values
        df1[f+'_median'] = df_file[f].median().values
        df1[f+'_std'] = df_file[f].std().values        
        df1[f+'_ptp'] = df1[f+'_max'] - df1[f+'_min']
    return df1

df_train = get_features(train)
df_train


# In[10]:


df_test = get_features(test)
df_test


# In[11]:


df_train.to_pickle('./df_train.pkl')
df_test.to_pickle('./df_test.pkl')


# In[12]:


import pickle
with open('./df_train.pkl', 'rb') as file:
    df_train = pickle.load(file)

with open('./df_test.pkl', 'rb') as file:
    df_test = pickle.load(file)


# In[13]:


import lightgbm as lgb
clf = lgb.LGBMClassifier(
    num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='multiclass',
    max_depth=-1, learning_rate=0.005, min_child_samples=3, random_state=2021,
    n_estimators=2000, subsample=1, colsample_bytree=1)

clf.fit(df_train.drop(['file_id','label'], axis=1), df_train['label'])


# In[14]:


# ' '.join 用 ' '做拼接间隔符
result = clf.predict_proba(df_test.drop('file_id', axis=1))
result


# In[15]:


result = pd.DataFrame(result, columns=['prob0','prob1','prob2','prob3','prob4','prob5','prob6','prob7'])
result['file_id'] = df_test['file_id'].values
result


# In[16]:


columns = ['file_id', 'prob0','prob1','prob2','prob3','prob4','prob5','prob6','prob7']
result.to_csv('./baseline_lgb_2000_file_id.csv', index=False, columns=columns)


# In[17]:


result = pd.read_csv('./baseline_lgb_2000_file_id.csv')
result

