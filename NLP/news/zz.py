#--*encoding=utf8-*-
import jieba
import re
jieba.load_userdict('dic.txt')
fp=open('cancer.txt','r',encoding='utf8')
fout=open('output.txt','w',encoding='utf8')

def merge_two_list(a, b):
    c=[]
    len_a, len_b = len(a), len(b)
    minlen = min(len_a, len_b)
    for i in range(minlen):
        c.append(a[i])
        c.append(b[i])

    if len_a > len_b:
        for i in range(minlen, len_a):
            c.append(a[i])
    else:
        for i in range(minlen, len_b):
            c.append(b[i])  
    return c

#regex1=u'(?:[^\u4e00-\u9fa5（）*&……%￥$，,。.@! ！]){1,5}期'
#regex2=r'(?:[0-9]{1,3}[.]?[0-9]{1,3})%'
regex1=
regex2=
p1=re.compile(regex1)
p2=re.compile(regex2)

for line in fp.readlines():
    result1=p1.findall(line)
    if result1:       
        regex_re1=result1
        line=p1.sub("FLAG1",line)
    result2=p2.findall(line)
    if result2:
        line=p2.sub("FLAG2",line)
    words=jieba.cut(line)
    result="/".join(words)
    if "FLAG1" in result:
        result=result.split("FLAG1")
        result=merge_two_list(result,result1)
        result="//".join(result)
    if "FLAG2" in result:       
        result=result.split("FLAG2")
        result=merge_two_list(result,result2)
        result="//".join(result)        
    print(result)
    fout.write(result)
fout.close()
