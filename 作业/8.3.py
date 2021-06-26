#_*_ coding:utf-8_*_
class Calculator():
    def __init__(self, first, second):
        self.first=first
        self.second=second

    def add(self):
        value=self.first+self.second
        print('%0.2f+%0.2f=%0.2f'%(self.first,self.second,value))


    def reduce(self):
        value=self.first-self.second
        print('%0.2f-%0.2f=%0.2f'%(self.first,self.second,value))

    def mult(self):
        value=self.first*self.second
        print('%0.2f*%0.2f=%0.2f'%(self.first,self.second,value))

    def divide(self):
        if self.second==0:
            print('分母不能为0！')
        else:
            value = self.first / self.second
            print('%0.2f/%0.2f=%f' % (self.first, self.second, value))

if __name__ == '__main__':
    Calculator(3, 5).add()
    Calculator(3, 5).reduce()
    Calculator(3, 5).mult()
    Calculator(3,0).divide()