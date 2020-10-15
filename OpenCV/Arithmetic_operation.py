import numpy as np
import cv2
x=np.uint8([250])
y=np.uint8([10])
#OpenCV的加法是一种饱和操作，numpy的加法是一种模操作
print(cv2.add(x,y)) #260>255=255
print(x+y) #260%255=4


#图像混合 需要注意的是两幅图像大小要一致
# 两幅画的权重不同，给人一种混合或透明的感觉，图像混合计算公式：
# g(x) = (1−α)f0 (x)+αf1 (x)    dst = α·img1 + β·img2+γ
img1=cv2.imread('1.jpg')
print(img1.size)
img2=cv2.imread('2.jpg')
print(img2.size)
dst = cv2.addWeighted(img1,0.7,img2,0.3,0)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()



