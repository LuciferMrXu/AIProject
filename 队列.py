#_*_ coding:utf-8_*_
# from pythonds.basic.queue import Queue
# from pythonds.basic.deque import Deque
'''
队列：队列是有序集合，新添加的一端为队尾，另一端为队头。
当一个元素从队尾进入队列时，一直向队首移动，直到它成为移除的元素为止。
这种排序是先进先出FIFO
'''
class Queue():
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return self.items == []
    def enqueue(self, item):
        self.items.insert(0,item)
    def dequeue(self):
        return self.items.pop()
    def size(self):
        return len(self.items)

'''
双端队列，是与队列类似的项的有序集合，有两个端部(首部和尾部)，可以在两端添加新项和删除。
这种混合的线性结构提供单个栈和队列的所有能力
'''

class Deque:
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return self.items == []
    def addFront(self, item):
        self.items.append(item)
    def addRear(self, item):
        self.items.insert(0,item)
    def removeFront(self):
        return self.items.pop()
    def removeRear(self):
        return self.items.pop(0)
    def size(self):
        return len(self.items)

'''
小游戏：小时候大家玩过一个游戏，大家围成一个圈，从某一个人开始报数，
知道某个人报数(报的数之前定义好的数值如30、50等等，之后给这个人一定的惩罚措施）
'''
def game(namelist,num):
    queue = Queue()
    '''
    把名单列表里面的名字全部添加到队列中
    '''
    for name in namelist:
        queue.enqueue(name)

    while queue.size()>1:
        '''
        循环队列定义好某几个数从队列头中进行删除操作，被删除之后添加到队列尾中
        '''
        for i in range(num):
            queue.enqueue(queue.dequeue())
        queue.dequeue()

    return queue.dequeue()

'''
判断一个str是不是回文数
'''
def check(datastr):
    biqueue = Deque()
    '''
    遍历字符串加入到队列中
    '''
    for ch in datastr:
        biqueue.addRear(ch)

    Equal = True
    '''
    最终队列剩下的个数为0或者1为成功
    '''
    while biqueue.size()>1 and Equal:
        first = biqueue.removeFront()
        last = biqueue.removeRear()
        '''
        队列前面取一个，后面取一个，若二者不相等则不匹配，返回False
        '''
        if first != last:
            Equal = False

    return Equal

if __name__=="__main__":
    print(game(['hefei','shanghai','hangzhou','shenzhen','guangzhou'],10))
    print(check('123321'))
