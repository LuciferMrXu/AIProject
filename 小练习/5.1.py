#_*_ coding:utf-8_*_
def Screen():
    list1 = ['打火机', '爆竹', '汽油', '乙醇', '硝化甘油', '黑火药', '管制刀具']
    li=Cycle()
    for i in li:
        if i in list1:
            print('携带危险物品，禁止上车！')
            break
    else:
        print('请上车！')

def Cycle():
    n = 0
    list1=[]
    while True:
        if n == 0:
            pass
        elif n == 1:
            break
        a = input('请输入携带的物品：\n')
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

if __name__ == '__main__':
    Screen()







