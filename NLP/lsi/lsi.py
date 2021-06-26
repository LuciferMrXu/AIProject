# -*- coding: UTF-8 -*-
from gensim import corpora,similarities,models
import jieba
stoplist = [ line.strip() for line in open("stop_words_zh_UTF-8.txt",'r',encoding="utf8") ]
lines=open("sentences.txt",'r',encoding="utf8").readlines()
corpora_sen = []
for line in lines:
    words=jieba.cut(line.replace("\n","").strip())#分词
    seg=[word for word in list(words) if word not in stoplist]#去掉停用词
    corpora_sen.append(seg)

# 生成字典
dictionary = corpora.Dictionary(corpora_sen)
#dictionary.dfs
#把句子转化为向量
corpus=[]
for word in corpora_sen:
    temp=dictionary.doc2bow(word)
    corpus.append(temp)
#len(corpus)
tfidf_model = models.TfidfModel(corpus)
corpus_tfidf = tfidf_model[corpus]
#使用lsi模型相似度计算
lsi=models.LsiModel(corpus_tfidf)#把tfidf语料投射到lsi空间
#lsi.print_topics(num_words=10)

copus_lsi=lsi[corpus_tfidf]#把tf-idf空间向量转化为lsi空间向量
#print(copus_lsi.corpus.corpus)
#len(copus_lsi.corpus.corpus)
print(copus_lsi.corpus.corpus)
similarity_lsi = similarities.Similarity('Similarity-LSI-index', copus_lsi, num_features=600,num_best=3)#num_best=5输出相似度前5的句子
test_sen = '自然语言处理与人工智能'
seg_sen = list(jieba.cut(test_sen)) 
test_corpus = dictionary.doc2bow(seg_sen)  
lsi_vector=lsi[test_corpus]
print(similarity_lsi[lsi_vector])
#[(19, 0.71578741073608398), (18, 0.66015541553497314), (22, 0.45858818292617798), (21, 0.23949787020683289), (20, 0.16593065857887268)]
#(19, 0.71578741073608398)意思是索引号为19的句子和测试句子相似度为0.71578741073608398