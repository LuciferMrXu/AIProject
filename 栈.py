# -- encoding:utf-8 --
# from pythonds.basic.stack import Stack
'''
python实现栈：栈是一个有序集合，其中添加和删除元素都是发生在同一端，通常称作发生操作的这一端为顶部，对应的端为底部
例子说明：一个桶里面装很多东西，后放进里面的，先拿出来（也叫后进先出LIFO）
'''
class Stack():
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return self.items == []
    def push(self,item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def peek(self):
        return self.items[len(self.items)-1]
    def size(self):
        return len(self.items)

'''
括号匹配问题
'''
def matches(open,close):
    opens = '([{'
    closes = ')]}'
    return opens.index(open) == closes.index(close)

def check(dataStr):
    s = Stack()
    balanced = True
    index = 0

    while index < len(dataStr) and balanced:
        data = dataStr[index]
        if data in '([{':
            s.push(data)
        else:
            if s.isEmpty():
                balanced = False
            else:
                s.pop()
        index = index + 1
    if balanced and s.isEmpty():
        return True
    else:
        return False

'''
十进制数转换成其他不同进制数的问题
例如 给一个十进制数8  
8%2=0   8//2=4  4%2=0  4//2=2  2%2=0 2//2=1 1%2=1 1//2=0, 二进制为1000(从后向前取)
'''

def convert(num,base):
    digits = '0123456789ABCDEF'
    stack = Stack()
    while num > 0:
        stack.push(num % base)
        num = num // base
    newStr = ''
    while not stack.isEmpty():
        newStr = newStr + digits[stack.pop()]
    return newStr




if __name__=="__main__":
    print(check('[(])'))
    print(convert(1024,2))