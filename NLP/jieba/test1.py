#--*encoding=utf8-*-8
import sys
import os
import jieba



sent='在包含问题的所有解的解空间数中，按照深度优先搜索的策略，从根节点出发深度探索解空间树。'

# 全模式
wordlist=jieba.cut(sent,cut_all=True)
print('|'.join(wordlist))

# 精确切分
wordlist=jieba.cut(sent,cut_all=False)
print('|'.join(wordlist))

# 搜素引擎模式
wordlist=jieba.cut_for_search(sent)
print('|'.join(wordlist))

# 用户词典
jieba.load_userdict('dict1.txt')
wordlist=jieba.cut(sent)
print('|'.join(wordlist))