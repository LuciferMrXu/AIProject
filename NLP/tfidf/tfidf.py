#-*- coding=utf8 -*-
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
import jieba
import numpy as np

def tf_idf_matrix(cut_word_file):
    corpus=[]
    tfidfdict={}
    f_res = open('sk_tfidf.txt', 'w',encoding="utf8")
    tfidf_file=open('tfidf.txt','w',encoding="utf8")
    lines=open(cut_word_file,'r',encoding="utf8").readlines()
    for line in lines:
        line=line.replace('\n',' ')
        corpus.append(line)
    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵.
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值  
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语  
    weight=tfidf.toarray()# 
    for i in range(len(weight)):#文本个数,output.txt文件中一行为一个文本
        tempdict={}
        list1=[]
        sss=''
        for j in range(len(word)): #所有单词数           
            if weight[i][j] == np.float64(0):
                continue
            else :
                #sss=word[j]+':'+str(weight[i][j])
                #list1.append(sss)
                #xx=word[j].decode('gbk')
                if word[j] in tfidfdict.keys():  #更新全局TFIDF值
                    tfidfdict[word[j]] += weight[i][j]
                else:
                    tfidfdict.update({word[j]:weight[i][j]})
                sorted_tfidf = sorted(tfidfdict.items(), 
                                      key=lambda d:d[1],  reverse = True )               
                tempdict[word[j]]=weight[i][j]
        sort_dict=sorted(tempdict.items(),key=lambda item:item[1],reverse=True)
        #sort_dict=sorted(tempdict.items(),key=lambda item:item[1],reverse=True)
        if sort_dict:
            str4=''
            for key,value in sort_dict:
                if value>0.15:
                    str4+=key+':'+str(value)+'@@'              
            str5=lines[i].replace(' ','').replace('\n','')+'&&&&&&'+str4+'\n' 
            #str5=lines[i].replace(' ','').replace('\n','')+'&&&&&&'+sss+'\n' 
            tfidf_file.write(str5)
            str5=''
    tfidf_file.close()
    for i in sorted_tfidf:  #写入文件
        f_res.write(i[0] + '\t' + str(i[1]) + '\n')      

if __name__=='__main__':
    jieba.load_userdict('../data/dic.txt')
    fp=open('../data/cancer.txt','r',encoding='utf8')
    fout=open('output.txt','w',encoding='utf8')
    for line in fp.readlines():
        words=jieba.cut(line)
        result=' '.join(words)
        fout.write(result)
    fout.close()    
    tf_idf_matrix("output.txt")