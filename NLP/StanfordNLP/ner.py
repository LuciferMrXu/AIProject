#-*-encoding=utf8-*-

from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('G:\\NLP\\StanfordNLP\\stanfordnlp', lang='zh')

fin=open('news.txt','r',encoding='utf8').readlines()
fout=open('result.txt','w',encoding='utf8')

for line in fin:
    if line:
        line=line.replace("\n","")
        line=line.strip()
        print(nlp.word_tokenize(line)) # 分词
        print(nlp.parse(line))      # 语法分析
        print(nlp.dependency_parse(line))  # 语法依赖关系
        ner=nlp.ner(line)    # 命名实体识别
        tag=nlp.pos_tag(line)  # 词性
        strNer=''
        strTag=''
        for each in ner:
            if len(each)==2:
                temp= each[0]+"/"+each[1]+" "
                strNer+=temp
        for each in tag:
            if len(each)==2:
                temp= each[0]+"/"+each[1]+" "
                strTag+=temp    
        fout.write(strNer+'\n'+strTag+'\n')
fout.close()   
    
