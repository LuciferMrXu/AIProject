#_*_ coding:utf-8_*_
store = []
def add_student(a):
    store.append(a)
    i=1
    for j in store:
        print('同学%s:%s\n'%(i,j))
        i+=1

def del_student(a):
    try:
        b=int(a)
    except Exception :
        print('请输入数字！')
    else:
        try:
            print('%s同学对不起，您被开除了！'%store.pop(b-1))
        except Exception:
            print('请确保该同学序号在上表中！')
    n=1
    for m in store:
        print('同学%s:%s\n' % (n,m))
        n += 1

if __name__=='__main__':
    while True:
        add=input('请输入学生姓名，输入完毕请按#号键结束！\n')
        if add=='#':
            break
        add_student(add)


    n=0
    while True:
        if n == 0:
            pass
        elif  n == 1:
            break
        delete=input('请输入被删除学生的序号，删除完毕请按#结束！\n')
        if delete=='#':
            break
        del_student(delete)
        while True:
            m = input('是否继续输入(y/n):\n')
            if m == 'y':
                break
            elif m == 'n':
                n = 1
                break
            else:
                print('请重新输入！')

