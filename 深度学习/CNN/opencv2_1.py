# -- encoding:utf-8 --
import cv2
import numpy as np
import matplotlib.pyplot as plt
#raw 
img_src = cv2.imread('data\picture1.png')
# cv2.namedWindow('input_image',cv2.WINDOW_AUTOSIZE)
# cv2.imshow('input_image',img_src)
# cv2.waitKey(0)
# openCV通道是BGR，需要转换成默认的RGB
img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

#平移
'''
[[1 0 tx]    tx是水平平移量 ty是竖直平移量
 [0 1 ty]]
'''
H = np.float32([[0,1,50],[1,0,100]])
rows,cols = img_src.shape[:2]
res = cv2.warpAffine(img_src,H,(rows,cols)) #需要图像、变换矩阵、变换后的大小
#cv2.imshow('move_image',res)
#cv2.waitKey(0)
res_plt = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
plt.imshow(res_plt)
plt.show()


#缩放
res1 = cv2.resize(img_src,None,fx=0.7,fy=1.2,interpolation=cv2.INTER_CUBIC)
#cv2.imshow('resize_image1',res1)
#cv2.waitKey(0)
res1_plt = cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)
plt.imshow(res1_plt)
plt.show()
#直接规定缩放大小，这个时候就不需要缩放因子
height,width = img_src.shape[:2]
res2 = cv2.resize(img_src,(2*width,1*height),interpolation=cv2.INTER_CUBIC)
#cv2.imshow('resize_image2',res2)
#cv2.waitKey(0)
res2_plt = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)
plt.imshow(res2_plt)
plt.show()

#旋转 
rows,cols = img_src.shape[:2]
#第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
M = cv2.getRotationMatrix2D((rows/2,cols/2),45,0.5)
#第三个参数：变换后显示的区域大小，包含填充区域
res3 = cv2.warpAffine(img_src,M,(2*rows,cols))
#cv2.imshow('rot_image',res3)
#cv2.waitKey(0)
res3_plt = cv2.cvtColor(res3, cv2.COLOR_BGR2RGB)
plt.imshow(res3_plt)
plt.show()

#灰度变换——提取轮廓信息
gray_img=cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_image',gray_img)
cv2.waitKey(0)

#色彩空间转换
hsv_img = cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)
cv2.imshow('hsv_img',)
cv2.waitKey(0)
yuv_img = cv2.cvtColor(img_src,cv2.COLOR_BGR2YUV)
cv2.imshow('yuv_image',yuv_img)
cv2.waitKey(0)

cv2.destroyAllWindows()