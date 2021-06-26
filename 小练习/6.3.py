#_*_ coding:utf-8_*_
def Cycle():
    n = 0
    list1=[]
    while True:
        if n == 0:
            pass
        elif n == 1:
            break
        a = input('请输入列表元素：\n')
        list1.append(a)
        while True:
            e = input('是否继续输入(y/n):\n')
            if e == 'y':
                break
            elif e == 'n':
                n = 1
                break
            else:
                print('请重新输入！')

    return list1

def main():
    li=Cycle()
    print(li)
    try:
        a=int(input('请输入想要修改第几个元素：\n'))
    except Exception:
        print('请输入数字！')
    else:
        b=input('请输入想要修改的内容：\n')
        li[a-1]=b
        print(u'修改后的列表为',li)

if __name__=='__main__':
    main()