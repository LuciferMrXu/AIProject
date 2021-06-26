#_*_ coding:utf-8_*_
def calculator(m,i,n):
    if i=='+':
        return m+n
    elif i=='-':
        return m-n
    elif i=='*':
        return m*n
    elif i=='/':
        return m/n
    else:
        return '请输入四则运算符！'

def main():
    a=input('请输入第一位数字：\n')
    b=input('请输入运算符：\n')
    c=input('请输入第二位数字：\n')
    if a.isdigit() and c.isdigit():
        if b=='/' and c=='0':
            print('分母不能为0！')
        else:
            a=int(a)
            c=int(c)
            value = calculator(a, b, c)
            print('计算结果为%s' % value)
    else:
        print('请输入整数！')

if __name__=='__main__':
    main()