# -- encoding:utf-8 --
"""
Create by ibf on 2018/10/20
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer, TfidfVectorizer

arr1 = [
    "This is spark, spark sql a every good",
    "Spark Hadoop Hbase",
    "This is sample",
    "This is anthor example anthor example",
    "spark hbase hadoop spark hive hbase hue oozie",
    "hue oozie spark"
]
arr2 = [
    "this is a sample a example",
    "this c c cd is another another sample example example",
    "spark Hbase hadoop Spark hive hbase"
]
df = arr2

# 一、词袋法的使用
# a. 词袋法的sklearn对象的构建
"""
input='content': 指定对什么数据进行词袋法转换操作，content表示内存中的文本数据对象，'filename', 'file'分别表示从对应的文件中获取数据
encoding='utf-8', 字符集
stop_words=None, 指定有哪些单词是不能作为特征属性的 --> 称之为停止词
token_pattern=r"(?u)\b\w\w+\b", 指定单词所需要满足的正则字符串，
max_features=None, 最多允许的特征数目，None表示不限制
dtype=np.int64 特征属性的数据类型
"""
count = CountVectorizer(stop_words=['cd', 'is', 'this'])
# b.模型训练，相当于在训练数据中找哪些单词作为特征属性
df1 = count.fit_transform(df)
# c. 基于训练好的模型对测试数据做转换
r1 = count.transform(arr1).toarray()
print("特征属性单词:\n{}".format(count.get_feature_names()))
print(df1.toarray())
print(r1)

# 二、IDF操作(必须在TF操作之后进行)
tfidf = TfidfTransformer()
df2 = tfidf.fit_transform(df1)
r2 = tfidf.transform(r1)
print(df2.toarray())
print(r2.toarray())

# 三、HashTF
print("\n\n")
hashing = HashingVectorizer(n_features=10)
df3 = hashing.fit_transform(df)
r3 = hashing.transform(arr1)
print(df3.toarray())
print(r3.toarray())
