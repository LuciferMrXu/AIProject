#-*-coding=utf8-*-
from gensim import corpora,similarities,models
import jieba
import numpy as np
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
#使用LDA模型相似度计算
lda=models.LdaModel(corpus=corpus_tfidf ,id2word=dictionary,num_topics=10)#把corpus_tfidf语料投射到lda空间
#lda.print_topics(4)
lda.save('lda.model')#保存模型
lda=models.ldamodel.LdaModel.load('lda.model')#加载模型

test_sen = '自然语言处理与人工智能'
test_sen1 = ['欧盟量化宽松政策经验非常丰富','自然语言处理是人工智能的核心']
result=[]
test_vec_list=[]
seg_sen = list(jieba.cut(test_sen)) 
test_corpus = dictionary.doc2bow(seg_sen)  
test_lda=lda[test_corpus]
test_vec=[i[1] for i in test_lda]
for sentence in test_sen1:
    seg1= list(jieba.cut(sentence)) 
    test_corpus1 = dictionary.doc2bow(seg1)  
    seg_vec=lda[test_corpus1]
    weight = [i[1] for i in seg_vec]
    test_vec_list.append(weight)

try:
    for i in range(2):#np.dot(test_vec, test_vec_list[i])点乘
        sim = np.dot(test_vec, test_vec_list[i]) / (np.linalg.norm(test_vec) * np.linalg.norm(test_vec_list[i]))#np.linalg.norm(test_vec)向量平方根
        result.append(sim)
except ValueError:
    sim=0
print(result)
