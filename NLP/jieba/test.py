#--*encoding=utf8-*-
import jieba
import jieba.posseg as ps
import jieba.analyse as ana


s1=jieba.cut('中华民国的首都在南京',cut_all=True)
s11='/'.join(s1)
print(s11)

s2=jieba.cut('中华民国的首都在南京',cut_all=False)
s21='/'.join(s2)
print(s21)

s3=jieba.cut_for_search('中华民国的首都在南京')
s31='/'.join(s3)
print(s31)

s4=ps.cut('中华民国的首都在南京')
for word,tag in s4:
    print('word is:'+word+'and tag is:'+tag)


s5=jieba.tokenize('中华民国的首都在南京')    
for  s in s5:
    print('frist value is:%s,second value is:%s,third value is:%s.'%(s[0],s[1],s[2]))
