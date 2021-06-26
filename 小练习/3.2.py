#_*_ coding:utf-8_*_
store=[]
while True:
    a=input('请输入学生姓名，输入完毕请按#号键结束！\n')
    if a=='#':
        break
    else:
        store.append(a)
for j in store:
    print('%s同学，你好！'%j)
