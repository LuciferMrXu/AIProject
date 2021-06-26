#coding:utf-8
#https://github.com/hankcs/HanLP


from pyhanlp import *

print(HanLP.segment('回到过去重新再来过'))
testCases = [
    "买水果然后来世博园最后去世博会",
    "刘湖南是他的同学",
    "欢迎新老师生前来就餐",
    "上海自来水来自海上",
    "他多想回到过去"]
for sentence in testCases: print(HanLP.segment(sentence))
# 关键词提取
document = "昏天又暗地忍不住的流星" \
           "烫不伤被冷藏一颗死星" \
           "苦苦的追寻茫茫然失去" \
           "可爱的可恨的多可惜" \
           "梦中的梦中梦中人的梦中"

print(HanLP.extractKeyword(document, 2))
# 自动摘要
print(HanLP.extractSummary(document, 3))
# 依存句法分析
print(HanLP.parseDependency("天空是什么颜色的？"))

#jieba -> hanlp or ltp(规则模板、意图识别、情感分析)   ->gensim(语义理解sentence2vec) ->tensorflow(深度学习)