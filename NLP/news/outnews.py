#--*encoding=utf8-*-
import jieba
jieba.load_userdict('dic.txt')
fp=open('cancer.txt','r',encoding='utf8')
fout=open('output.txt','w',encoding='utf8')
for line in fp.readlines():
    words=jieba.cut(line)
    result='/'.join(words)
    fout.write(result)
fout.close()

