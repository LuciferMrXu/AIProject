#_*_ coding:utf-8_*_
import numpy as np
a=np.array([
    [1,1,0],
    [0,0,1],
    [1,1,1],
    [0,1,0],
    [0,0,1],
    [0,1,1],
    [0,0,0],
    [1,0,0]
])
u,sigma,v=np.linalg.svd(a)
print(u)
print(sigma)
print(v)
# 获取前k列，这里k=3
A=np.dot(np.dot(u[:,:3],np.diag(sigma)),v[:,:3].T)
print(A)
# k=2时
A=np.dot(np.dot(u[:,:2],np.diag(sigma)[:2,:2]),v[:,:2].T)
print(A)

# 计算相似度
def sim(a,b):
    m = a[0] * b[0] + a[1] * b[1]
    n1 = a[0] ** 2+a[1] ** 2
    n2 = b[0] ** 2 + b[1] ** 2
    n = np.sqrt(n1)*np.sqrt(n2)
    print(1.0*m/n)
    return 1.0*m/n

# 第一篇与第二篇相比
sim([-0.46561028, -0.66960599 ,-0.57864919],[ 0.58421721, 0.25857088, -0.76930576])
# 第一篇与第三篇相比
sim([-0.46561028, -0.66960599, -0.57864919],[0.66475357, -0.69625349, 0.27080208])