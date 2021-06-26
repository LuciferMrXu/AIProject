#_*_ coding:utf-8_*_
store=[]
i=1
while True:
    a=input('请输入学生姓名，输入完毕请按#号键结束！\n')
    if a=='#':
        break
    else:
        store.append('同学%s:%s\n'%(i,a))
        i+=1
j=1
while True:
    b=input('请输入老师姓名，输入完毕请按#号键结束！\n')
    if b=='#':
        break
    else:
        store.append('老师%s:%s\n'%(j,b))
        j+=1
c=input('请输入班主任姓名：\n')
store.insert(0,'班主任：%s\n'%c)

for k in store:
    print(k)