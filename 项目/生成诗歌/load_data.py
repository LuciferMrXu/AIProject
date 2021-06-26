# 数据读取、数据处理、Batch操作
import pandas as pd
import os
import collections
import re
data_dir='data/poetry.txt'
# 不仅要得到内容，而且还需要去除一些特殊的符合数据
def data_load():
    word_list=[]
    with open(data_dir,encoding='utf-8') as f:
        for item in f.readlines():
            item_split=item.split(':')
            if len(item_split)>1:
                contents=item.split(':')[1]
                # 异常符号的替换清晰
                if '】' in contents or '__' in contents:
                    continue
                contents=contents.replace(' ','').replace(')','').replace('(','')
                re_complile=re.compile('(（.*?）)')
                contents=re.sub(re_complile,'',contents)
                word_list.append('['+contents+']')
            # [:表示起始符   ]:表示终止符
    # 按照诗的字数排序
    poertrys = sorted(word_list, key=lambda line: len(line))
    return poertrys

# 构建字典
def get_vocab(poertrys):
    # 统计每个字出现的次数
    all_words=list(''.join(poertrys))
    counter=collections.Counter(all_words)
    counter_paris=sorted(counter.items(),key=lambda x:x[1])
    words,_=zip(*counter_paris)
    words=words+(' ',)
    vocab=dict(zip(words,range(len(words))))
    de_vocab={value:key for key,value in vocab.items()}
    return vocab,de_vocab

# 转换文字到字典下标的形式
def word2vec(vocab,poertrys):
    to_num=lambda word:vocab.get(word,len(poertrys))
    poetrys_vector=[list(map(to_num,poetry)) for poetry in poertrys]
    return poetrys_vector