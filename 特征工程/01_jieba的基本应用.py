# -- encoding:utf-8 --
"""
使用jieba的主要耗时就在于根据业务特征自定义词典
Create by ibf on 2018/10/20
"""

import jieba

# 一、基础分词
word_str = '上海自来水来自海上'
word_str = '梁国杨氏子九岁，甚聪惠。孔君平诣其父，父不在，乃呼儿出。为设果，果有杨梅。孔指以示儿曰：此是君家果。儿应声答曰：未闻孔雀是夫子家禽'
"""
def cut(self, sentence, cut_all=False, HMM=True)
 sentence: 给定需要分词的字符串
 cut_all: 指定是否进行全模式的划分，默认为False，表示为精确模型划分
 HMM: 指定是否开启HMM发生新词的规则
方法返回的对象是一个迭代器，不是集合，所以数据在遍历的时候只能遍历一次
"""
seg_a = jieba.cut(word_str, cut_all=True)
seg_b = jieba.cut(word_str)
seg_c = jieba.cut(word_str, HMM=False)
seg_d = jieba.cut_for_search(word_str)
print("全模式:[{}]".format('\t'.join(seg_a)))
print("精确模式:[{}]".format('\t'.join(seg_b)))
print("精确模式(HMM为False):[{}]".format('\t'.join(seg_c)))
print("搜索模式的分词:{}".format('\t'.join(seg_d)))

# 添加词典
word_str = '梁国杨氏子九岁,甚聪惠'
seg_a = jieba.cut(word_str)
print("原始的分词效果:[{}]".format('\t'.join(seg_a)))

# 方式一：在代码中临时添加词语
# """
# def add_word(self, word, freq=None, tag=None):
#   word: 是需要添加的单词
#   freq: 是需要添加单词的词频，词频越高，就表示该单词越有可能出现
#   tag: 是该单词的词性，eg: 名词(n)，动词(v) ......
# """
# jieba.add_word('杨氏')
# jieba.add_word('聪惠', freq=100)
# seg_b = jieba.cut(word_str)
# print("添加分词后的效果:[{}]".format('\t'.join(seg_b)))
# # 删除自定义的单词
# jieba.del_word('聪惠')
# print("删除自定义词典后的效果:[{}]".format('\t'.join(jieba.cut(word_str))))

# 方式二：自定义一个词典文件，然后将文件加载到当前运行环境中即可达到效果
jieba.load_userdict('./mydict.txt')
seg_b = jieba.cut(word_str)
print("添加自定义词典后的效果:[{}]".format('\t'.join(seg_b)))

print("\t".join(jieba.cut('上海自来水来自海上')))
