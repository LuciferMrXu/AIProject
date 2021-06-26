# -- encoding:utf-8 --
"""
Create by ibf on 2018/10/21
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
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
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=28, train_size=0.7)

# 3. 数据的特征工程
# 词向量转换
tfidf = TfidfVectorizer()
x_train = tfidf.fit_transform(x_train['jieba_cut_content'].astype('str'))
x_test = tfidf.transform(x_test['jieba_cut_content'].astype('str'))

# 词向量降维
svd = TruncatedSVD(n_components=20, random_state=28)
x_train = svd.fit_transform(x_train)
x_test = svd.transform(x_test)

# 4. 模型的构建
print("训练数据集大小:{}".format(x_train.shape))
print("测试数据集大小:{}".format(x_test.shape))
algo = LogisticRegression(penalty='l2', C=0.5, max_iter=1000)
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
