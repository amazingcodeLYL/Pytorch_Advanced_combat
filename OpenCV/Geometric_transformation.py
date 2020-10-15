import cv2
import numpy as np
from matplotlib import pyplot as plt
# img=cv2.imread('1.jpg')
###############################################
###################扩展缩放####################
# res=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
# #cv2.resize 参数：原图，输出图像尺寸，沿水平轴的比例因子，沿垂直轴的比例因子，插值法
# #cv2.INTER_AREA（使用像素关系重采样）、cv2.INTER_LINEAR（双线性插值）
# # cv2.INTER_CUBIC（立方插值）等变换方法，cv2.INTER_AREA适合缩小使用，cv2.INTER_LINEAR、cv2.INTER_CUBIC（慢）适合放大使用。
# height,width=img.shape[:2]
# res=cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)
# while(1):
#     cv2.imshow('res',res)
#     cv2.imshow('img',img)
#     if cv2.waitKey(1)&0xFF==27: #按ESC退出
#         break
# cv2.destroyAllWindows()
##############################################

############################################
##########旋转
# img=cv2.imread('1.jpg',0)
# # rows,cols=img.shape
# # M=cv2.getRotationMatrix2D((cols/2,rows/2),45,0.6)
# # #getRotationMatrix2D 参数:旋转的中心点，旋转角度，图像缩放因子
# # dst=cv2.warpAffine(img,M,(2*cols,2*rows))#仿射变换
# # #warpAffine 参数:输入图像，变换矩阵，输出图像大小，插值方法的组合，边界像素模式，边界填充值
# #
# # while(1):
# #     cv2.imshow('img',dst)
# #     if cv2.waitKey(1)==27:
# #         break
# # cv2.destroyAllWindows()
#################################################
img=cv2.imread('1.jpg')
rows, cols, ch = img.shape

# pts1 = np.float32([[50,50],[200,50],[50,200]])
# pts2 = np.float32([[10,100],[200,50],[100,250]])
#@仿射变换
# M = cv2.getAffineTransform(pts1, pts2)
# #getAffineTransform由三对点计算仿射变换 参数：输入图像的三角形顶点坐标，输出三角形顶点坐标
# dst = cv2.warpAffine(img,M,(cols, rows))
#透视变换
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M=cv2.getPerspectiveTransform(pts1,pts2)
dst=cv2.warpPerspective(img,M,(300,300))
plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()
#################################
######透视变换
