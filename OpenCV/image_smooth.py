import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('3.jpg')
#2D卷积
# kernel=np.ones((5,5),np.float32)/25
# det=cv2.filter2D(img,-1,kernel)
# #cv2.filter2D的第二个参数-1表示输出图像与输入图像有相同的深度
# plt.subplot(121),plt.imshow(img),plt.title('original')
# # plt.subplot(122),plt.imshow(det),plt.title('averaging')
# plt.show()

#平均
# blur=cv2.blur(img,(5,5)) #全部模糊
#高斯模糊
blur=cv2.GaussianBlur(img,(5,5),0)
#高斯滤波在去除图像中的高斯噪声方面非常有效。指定的高度和宽度必须为正奇数
#中值模糊
# blur=cv2.medianBlur(img,5)
#双边滤波
# blur=cv2.bilateralFilter(img,9,75,75)
# 边滤波在同时使用空间高斯权重和灰度值相似性高斯权重
# 在保持边界清晰的情况下有效的去除噪音，但比较慢
while(1):
    cv2.imshow('image',img)
    cv2.imshow('blur',blur)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
cv2.destroyAllWindows()
