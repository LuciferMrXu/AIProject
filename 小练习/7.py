#_*_ coding:utf-8_*_
# def function(n):
#     if n==1:
#         return 1
#     else:
#         return n*function(n-1)
# if __name__=='__main__':
#     try:
#         a=int(input('请输入n的值：\n'))
#         print(function(a))
#     except Exception:
#         print('请输入数字！')


# def main(n):
#     if isinstance(n,str):
#         return len(n)
#     else:
#         return '请输入字符串！'
# if __name__ == '__main__':
#     print(main(12344))



# n=0
# for i in range(1,5):
#     for j in range(1,5):
#         for m in range(1,5):
#             if i!=j and i!=m and j!=m:
#                 print(100*i+10*j+m)
#                 n+=1
# print(n)




# def reward(n):
#     if 0<n<=100000:
#         return n*0.1
#     elif 100000<n<=200000:
#         return 100000*0.1+(n-100000)*0.075
#     elif 200000<n<400000:
#         return 100000*0.1+100000*0.075+(n-200000)*0.05
#     elif 400000<n<=600000:
#         return 100000 * 0.1 + 100000 * 0.075 + 200000* 0.05+(n-400000)*0.03
#     elif 600000<n<=1000000:
#         return 100000 * 0.1 + 100000 * 0.075 + 200000* 0.05+200000*0.03+(n-600000)*0.015
#     elif n>1000000:
#         return 100000 * 0.1 + 100000 * 0.075 + 200000 * 0.05 + 200000 * 0.03 + 400000 * 0.015+(n-1000000)*0.01
#     else:
#         return '请输入正数!'
# if __name__=='__main__':
#     try:
#         a=int(input('请输入当月利润：\n'))
#         print('应发奖金数为：',reward(a))
#     except Exception:
#         print('请输入数字！')
















# i=0
# list1=[]
# while i<=35:
#     list1.append(i**2)
#     i+=1
# for j in range(0,1001):
#     if j+100 in list1:
#         if j+168 in list1:
#             print(j)





# def function(a,b):
#     if b==1:
#         S=a
#         return S
#     else:
#         S=(10*function(a,b-1)+a)
#         return S
# if __name__=='__main__':
#     S=0
#     try:
#         a=int(input('请输入a的值：\n'))
#         n = int(input('请输入n的值：\n'))
#     except Exception:
#         print('请输入数字！')
#     else:
#         for b in range(1,n+1):
#             S+=function(a,b)
#         print(S)





