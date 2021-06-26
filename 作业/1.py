#_*_ coding:utf-8_*_
def receive():
    a = input('请输入姓名：\n')
    b = input('请输入年龄：\n')
    return a,b

def output():
    while True:
        c,d=receive()
        if d.isdigit():
            print('我叫%s，今年%s岁！'%(c,d))
            break
        else:
            print('输入的年龄有误,请重新输入！')

if __name__=='__main__':
    output()