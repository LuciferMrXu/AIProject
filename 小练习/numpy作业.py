#_*_ coding:utf-8_*_
import numpy as np

# # 四维数组以后两位的维度为单位,计算里面元素的和
# list1=[[
#     [
#         [2,7,9,7],
#         [6,6,8,2],
#         [0,0,9,3]
#     ],
#     [
#         [5,4,1,4],
#         [5,7,9,7],
#         [8,4,1,4]
#     ]
# ]]
# arr1=np.array(list1)
# arr2=arr1.reshape(1,2,-1)
# print(arr2.sum(axis=2))


# # 在数组中相邻元素中插入两个0
# arr1=np.array([1,2,3,4,5])
# arr2=np.array([0,0,0,0,0])
# arr3=np.array([0,0,0,0,0])
# arr4=np.vstack((arr1,arr2,arr3))
# arr5=arr4.T.ravel()
# arr6=np.delete(arr5,(13,14))
# print(arr6)


# # 二维矩阵与三维矩阵相乘
# # arr1=np.ones((3,3,2))
# # arr2=np.array(list('222222222'),dtype='float64').reshape(3,3)
# # arr3=arr1.T*arr2
# # print(arr3.T)



# # 交换矩阵的两行
# arr=np.arange(25).reshape(5,-1)
# arr[[0,1]] = arr[[1,0]]
# print(arr)






# arr = np.zeros((10,10))
# arr[[0,-1],:]=1
# for i in range(1,10):
#     arr[np.ix_([i],[0,-1])]=1
# print(arr)




