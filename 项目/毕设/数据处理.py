#_*_ coding:utf-8_*_
from pymongo import MongoClient
import pandas as pd
import numpy as np
import jieba
import lda
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from PIL import Image,ImageSequence
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']    # 设置字体
mpl.rcParams['axes.unicode_minus'] = False      # 中文显示
from sklearn.cluster import KMeans


client = MongoClient('localhost', 27017)  # 连接数据库
db = client.mydb
collection1 = db.微博
collection2 = db.微博信息


def clean(data1,data2,school='重庆大学图书馆'):
    data1 = pd.DataFrame(list(data1.find()))   # 合并两张表
    data1 = data1.drop(['_id', '篇数'], axis=1)
    data2 = pd.DataFrame(list(data2.find()))
    data2 = data2.drop(['_id'], axis=1)
    data2 = data2.rename(columns={'关注': '总关注'})

    data=pd.merge(data1,data2,on='高校',how='inner')  # 生成层次索引
    new_date=data.set_index(['高校','粉丝','总关注','是否认证','介绍'])

    df=new_date.loc[school]['正文'].reset_index()  # 提取具体哪所高校的数据
    X = df['正文']
    X = X.replace("", np.NAN)     # 过滤缺省值
    X = X.dropna(how='any', axis=0)
    return X

def TextMining(data,topic_num=28,show=False):
    result=[]
    for i in data:
        i = i.replace(' ', '')
        document_cut = jieba.cut(i,cut_all=False,HMM=True)     # jieba分词
        result.append( ' '.join(document_cut))
    '''
        这里为什么用词袋法可以训练，tfid不行
    '''
    tfid =TfidfVectorizer(min_df=0, encoding='utf-8')
    weight = tfid.fit_transform(result)
    word = tfid.get_feature_names()
    print(weight)
    print(word)

    model=lda.LDA(n_topics=topic_num,n_iter=500,random_state=16)

    # 模型预测，得到文档-主题映射关系
    doc_topic = model.fit_transform(weight)
    print("大小:{}".format(doc_topic.shape))
    # 获取模型的主题词
    topic_word = model.topic_word_
    print("主题词数量: {}".format(topic_word.shape))

    '''
        np.argsort => 对当前主题中各个单词的频率按照从小到大排序，返回索引值
        np.array(vocab)[np.argsort(topic_dist)] => 获取从小到大排序后的单词(频率/概率)
        np.array(vocab)[np.argsort(topic_dist)][:-(n + 1):-1] => 获取最后的n个单词
    '''
    # 每个主题中的前7个单词

    words=[]
    n = 5
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(word)[np.argsort(topic_dist)][:-(n + 1):-1]
        print('*主题 {}\n- {}'.format(i, ' '.join(topic_words)))
        words.append(' '.join(topic_words))

    if show:
        # 计算每个主题中单词权重分布情况：
        plt.figure(figsize=(8, 9))
        # f, ax = plt.subplots(5, 1, sharex=True)
        for i, k in enumerate([1, 2, 3, 4, 5]):
            ax = plt.subplot(5, 1, i + 1)
            ax.plot(topic_word[k, :], 'r-')
            ax.set_xlim(-50, 50500)  # [0,4258]
            ax.set_ylim(0, 0.08)
            ax.set_ylabel(u"概率")
            ax.set_title(u"主题 {}".format(k))
        plt.xlabel(u"词", fontsize=14)
        plt.tight_layout()
        # plt.suptitle(u'主题的词分布', fontsize=18)
        plt.subplots_adjust(top=0.9)
        plt.show()

        plt.figure(figsize=(8, 9))
        for i, k in enumerate([1, 2, 3, 4, 5]):
            ax = plt.subplot(5, 1, i + 1)
            ax.stem(doc_topic[k, :], linefmt='g-', markerfmt='ro')
            ax.set_xlim(-1, topic_num + 1)
            ax.set_ylim(0, 1)
            ax.set_ylabel(u"概率")
            ax.set_title(u"文档 {}".format(k))
        plt.xlabel(u"主题", fontsize=14)
        #plt.suptitle(u'文档的主题分布', fontsize=18)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    return doc_topic,words

def kmeans(data,words,topic_num=28):
    algo = KMeans(n_clusters=topic_num, n_init=10)
    algo.fit_transform(data)
    con=pd.Series(algo.labels_)
    result=con.value_counts()

    value=[]
    fin_key=[]
    fin_value=[]
    for i in range(len(words)):
        value.append(result[i])
        word=words[i].split()
        for j in word:
            fin_key.append(j)
            fin_value.append(value[i])
    dic = dict(zip(fin_key, fin_value))
    print(dic)

    image = Image.open('back.png')  # 作为背景形状的图
    graph = np.array(image)    #参数分别是指定字体、背景颜色、最大的词的大小、使用给定图作为背景形状
    wc = WordCloud(font_path = 'simhei.ttf',
                   background_color = 'black',
                   # max_words = 100,
                   # max_font_size=165,
                   # min_font_size=75,
                   relative_scaling=True,
                   collocations=True,
                   mask = graph,
                   prefer_horizontal = 0.75
                    )
    wc.generate_from_frequencies(dic)#根据给定词频生成词云
    image_color = ImageColorGenerator(graph)
    plt.imshow(wc)
    plt.axis("off")#不显示坐标轴
    plt.show()

def main():
    X=clean(collection1,collection2)
    vector,words=TextMining(X)
    kmeans(vector,words)

main()