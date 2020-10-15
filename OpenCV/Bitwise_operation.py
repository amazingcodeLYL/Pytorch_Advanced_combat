import cv2
import numpy as np
#按位运算操作有:AND,OR,NOT,XOR...
img1=cv2.imread('1.jpg')
img2=cv2.imread('2.jpg')

rows,cols,channels=img2.shape
roi=img1[0:rows,0:cols]

img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#cv2.threshold阈值处理 参数：源图片，阈值0-255，填充色0-255，阈值类型
ret,mask=cv2.threshold(img2gray,175,255,cv2.THRESH_BINARY)
mask_inv=cv2.bitwise_not(mask)

img1_bg=cv2.bitwise_and(roi,roi,mask=mask)
img2_fg=cv2.bitwise_and(img2,img2,mask=mask_inv)

dst=cv2.add(img1_bg,img2_fg)
img1[0:rows,0:cols]=dst
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()