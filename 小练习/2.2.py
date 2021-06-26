#_*_ coding:utf-8_*_
store=[]
i=1
while True:
    a=input('请进输入信息，输入完毕请按#号键结束！\n')
    if a=='#':
        break
    else:
        store.append('元素%s:%s\n'%(i,a))
        i+=1
for j in store:
    print(j)





# a=input('请输入学生姓名，用空格隔开：')
# lista=a.split(' ')
# lista.append('班主任姓名')
# b=input('请输入老师姓名，用空格隔开：')
# listb=b.split(' ')
# listb.extend(lista)
# for i in lista:
#     if i=='班主任姓名':
#         continue
#     print(i,'问候语')