# -- encoding:utf-8 --
"""
Create by ibf on 2018/10/21
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.externals import joblib

# 1. 读取数据
file_path = './result_process02'
df = pd.read_csv(file_path)
df = df.dropna(axis=0)
# df.info()

# 2. 数据的分割
# TODO: 因为6万多邮件对于我python环境压力有点大，所以使用部分数据来训练模型
x = df.drop('label', axis=1)
y = df['label']

# 3. 数据的特征工程
# 词向量转换
x_train, _, _, _ = train_test_split(x, y, random_state=28, train_size=0.7)
tfidf = TfidfVectorizer()
x_train = tfidf.fit_transform(x_train['jieba_cut_content'].astype('str'))
content_vector_maxtrix = tfidf.transform(df['jieba_cut_content'].astype('str'))

# 词向量降维
svd = TruncatedSVD(n_components=20, random_state=28)
svd.fit(x_train)
content_vector_maxtrix = svd.transform(content_vector_maxtrix)

# 合并其它特征属性的数据
df01 = pd.DataFrame(content_vector_maxtrix)
df02 = df.drop('jieba_cut_content', axis=1)
df01_columns = df01.columns
df02_columns = df02.columns
df_final = pd.DataFrame(
    data=np.concatenate([np.array(df01), np.array(df02)], axis=1),
    columns=np.concatenate([df01_columns, df02_columns], axis=0))

# 数据划分
x = df_final.drop('label', axis=1)
y = df_final['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=28, train_size=0.7)
print("训练数据集大小:{}".format(x_train.shape))
print("测试数据集大小:{}".format(x_test.shape))

# 4. 模型的构建
algo = BernoulliNB(alpha=1.0,binarize=0.0005)
algo.fit(x_train, y_train)

# 5. 模型效果评估
y_train_pred = algo.predict(x_train)
y_test_pred = algo.predict(x_test)
print("训练数据集上的召回率:{}".format(recall_score(y_train, y_train_pred)))
print("训练数据集上的精确率:{}".format(precision_score(y_train, y_train_pred)))
print("训练数据集上的F1值:{}".format(f1_score(y_train, y_train_pred)))
print("测试数据集上的召回率:{}".format(recall_score(y_test, y_test_pred)))
print("测试数据集上的精确率:{}".format(precision_score(y_test, y_test_pred)))
print("测试数据集上的F1值:{}".format(f1_score(y_test, y_test_pred)))
