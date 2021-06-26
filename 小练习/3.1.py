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
for j in store:
    print(j)