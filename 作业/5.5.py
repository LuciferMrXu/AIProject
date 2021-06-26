#_*_ coding:utf-8_*_
a=int(input('请输入等腰三角形的高：'))
i=0
while i<a:
    n=0
    while n<=i:
        print('* ',end=' ')
        n += 1
    i+=1
    print()
i=a-1
while i>=1:
    n=1
    while i>=n:
        print('* ',end=' ')
        n+=1
    i-=1
    print()
