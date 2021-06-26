#_*_ coding:utf-8_*_
a=int(input('请输入RT三角形的边长：'))
for i in range(a):
    for j in range(i+1):
        print('* ', end=' ')
    print()