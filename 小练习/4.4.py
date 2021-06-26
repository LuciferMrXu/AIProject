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

    @classmethod
    def Get_Info(cls):
        for key, value in cls.vocabulary.items():
            print('单词：%s\n词性：%s\n解释：%s\n例句：%s' % (key, value['词性'], value['解释'], value['例句']))
            print()

if __name__=='__main__':
    n = 0
    while True:
        if n == 0:
            pass
        elif n == 1:
            break
        a = input('请输入单词：')
        b = input('请输入单词词性：')
        c = input('请输入单词解释：')
        d = input('请输入例句：')
        Dictionary(a,b,c,d)
        while True:
            e = input('是否继续输入(y/n):\n')
            if e == 'y':
                break
            elif e == 'n':
                n = 1
                break
            else:
                print('请重新输入！')

    Dictionary.Get_Info()
