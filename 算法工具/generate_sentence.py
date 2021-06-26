# 如何让计算机生成像人一样的话_
from icecream import ic
import random

grammar_rule = """
复合句子 = 句子 连词 复合句子 | 句子 
句子 = 主语s 谓语 宾语s
谓语 = 喜欢 | 讨厌 | 吃掉 | 玩 | 跑
主语s = 主语 和 主语 | 主语
宾语s = 宾语 和 宾语 | 主语
主语 = 冠词 定语 代号
宾语 = 冠词 定语 代号
代号 = 名词 | 代词
名词 = 苹果 | 鸭梨 | 西瓜 | 小狗 | 小猫 | 滑板 | 老张 | 老王
代词 = 你 | 我 | 他 | 他们 | 你们 | 我们 | 它
定语 = 漂亮的 | 今天的 | 不知名的 | 神秘的 | 奇奇怪怪的
冠词 = 一个 | 一只 | 这个 | 那个
连词 = 但是 | 而且 | 不过
"""


def parse_grammar(rule):
    grammar = dict()
    for line in rule.split('\n'):
        if not line.strip(): continue
        target, expand = line.split('=')
        expands = expand.split('|')

        grammar[target.strip()] = [e.strip() for e in expands]

    return grammar


def gene(target, grammar):
    if target not in grammar: return target

    expand = random.choice(grammar[target])

    return ''.join([gene(e, grammar) for e in expand.split()])


if __name__ == '__main__':
    another_rule = """
    expression = ( math ) op ( expression ) | math 
    math = num op num
    num = sing_num num | sing_num
    sing_num = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
    op = + | - | * | / | ^ | ->
    """
    for i in range(100):
        ic(gene('expression', parse_grammar(another_rule)))

