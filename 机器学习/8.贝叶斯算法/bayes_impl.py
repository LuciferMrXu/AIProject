from collections import defaultdict
from icecream import ic
'''
    贝叶斯垃圾邮件过滤
'''

class Bayes:
    def __init__(self) -> None:
        pass

    def __call__(self, trainTexts,predText):
        self.mian(trainTexts,predText)

    def train(self,text, ys):
        # 定义每个单词在不同y中出现的个数
        counts = defaultdict(int)
        # 定义y的个数
        yi_num = defaultdict(int)
        for line, yi in zip(text, ys):
            yi_num[yi] += 1
            for c in set(line):
                counts[(c, yi)] += 1

        # 若一个词没有出现过，则假设他出现的概率是句子长度分之一
        probs = defaultdict(lambda : 1/len(set(''.join(text))))
        # 注意：若一个字在句子中出现多次，也只统计一次。
        for c_y, t in counts.items():
            c, y = c_y
            probs[(c, y)] = counts[c_y] / yi_num[y]
        # probs是一句话属于类别y的概率 ；i在不同y下出现的概率
        return probs, {i: (yi_num[i] / len(ys)) for i in yi_num}


    def predict(self,query, evidence, hypothesis):
        pred = {}
        for yi in hypothesis:
            prod = 1
            for c in set(query):
                prod *= evidence[(c, yi)]

            pred[yi] = prod * hypothesis[yi]

        return pred

    def mian(self,trainTexts,predText):
        evidence, hypothesis = self.train([t for t, y in trainTexts], [y for t, y in trainTexts])
        ic(evidence)
        ic(self.predict(predText, evidence, hypothesis))


if __name__ == '__main__':
    texts = [
        ('今天天天天有天天最大的优惠！', 1),
        ('今天没有加班的话，赶快回家！', 0),
        ('不要等到明天！', 1),
    ]
    print(texts)
    predText = '今天没有优惠！'
    bayes = Bayes()
    ic(bayes(texts,predText))

