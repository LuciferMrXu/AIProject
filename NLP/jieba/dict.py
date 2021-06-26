#--*encoding=utf8-*-
import jieba
import jieba.posseg as ps
import sys
sys.path.append('./')

jieba.load_userdict('dic.txt')

s=jieba.cut("网易代理了暴雪的魔兽世界")
s1='  '.join(s)
print(s1)

s2=ps.cut("网易代理了暴雪的魔兽世界")
for word,tag in s2:
    print('word is:%s,and tag is:%s'%(word,tag))