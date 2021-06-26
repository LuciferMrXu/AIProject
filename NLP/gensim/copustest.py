# -*- coding: utf-8 -*-
from gensim import corpora,models
import jieba 
lines=open("news.txt","r",encoding="utf8").readlines()
corpus=[]
for line in lines:
    segs = list(jieba.cut(line))   
    corpus.append(segs)      
dictionary = corpora.Dictionary(corpus)
corpora1 = [ dictionary.doc2bow(text) for text in corpus ]
vec = [(0, 1), (4, 2)]
tfidf=models.TfidfModel(corpora1)
print(tfidf[vec])
print (dictionary)
print (dictionary.token2id)
print (corpora1)
print()