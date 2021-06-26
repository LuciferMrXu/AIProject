#_*_ coding:utf-8_*_
class Dictionary():
    vocabulary={}
    def __init__(self, word, part_of_speech, meaning, example_sentence):
        if word not in Dictionary.vocabulary:
            Dictionary.vocabulary[word]={'词性': part_of_speech, '解释': meaning, '例句': example_sentence}
            self.word=word
            self.part_of_speech=part_of_speech
            self.meaning=meaning
            self.example_sentence=example_sentence

    @staticmethod
    def check_vocabulary(word):
        if word in Dictionary.vocabulary:
            return True
        else:
            return False

    @classmethod
    def Get_Info(cls):
        word=input('请输入单词：')
        if Dictionary.check_vocabulary(word):
            print('%s单词词性为：%s\n解释：%s\n例句：%s\n'%(word,Dictionary.vocabulary[word]['词性'],Dictionary.vocabulary[word]['解释'],Dictionary.vocabulary[word]['例句']))
        else:
            print('该单词不存在！')

if __name__=='__main__':
    word1=Dictionary('Python','adj','想不出来','真心想不出来')
    word2=Dictionary('R','n','想不出来','是真的想不出来')
    word3=Dictionary('Go','adv','想不出来','不想了')
    Dictionary.Get_Info()