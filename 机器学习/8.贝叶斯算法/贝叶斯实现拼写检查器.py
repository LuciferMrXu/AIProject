#_*_ coding:utf-8_*_
import re
from collections import Counter

def words(text):
    txt=re.findall(r'\w+',text.lower())
    return txt

WORDS=Counter(words(open('./datas/belling_the_cat.txt').read()))

def P(word,N=sum(WORDS.values())):      # 统计词频（先验概率）
    return WORDS[word]/N


def correction(word):
    return max(candidates(word),key=P)


def candidates(word):
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def known(words):
    return set(w for w in words if w in WORDS)

'''
编辑距离：从一个字符串变成另一个字符串，增、删、改、查，每进行一步编辑距离加一
'''
def edits1(word):
    letters='abcdefghijklmnopqrstuvwxyz'
    splits=[(word[:i],word[i:]) for i in range(len(word)+1)]
    deletes=[L+R[1:] for L,R in splits if R]
    transposes=[L+R[1:]+R[0]+R[2:] for L,R in splits if len(R)>1]
    replaces=[L+c+R[1:] for L,R in splits if R for c in letters]
    inserts=[L+c+R for L,R in splits for c in letters]
    print(set(deletes+transposes+replaces+inserts))
    return set(deletes+transposes+replaces+inserts)


def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

if __name__=='__main__':
    word=''
    print(correction(word))
