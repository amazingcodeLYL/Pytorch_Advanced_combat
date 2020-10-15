import cv2
import numpy as np
#OpenCV检测程序的效率
img1=cv2.imread('1.jpg')
e1=cv2.getTickCount()#返回从操作系统启动到当前所经的计时周期数
for i in range(5,49,2):
    img1=cv2.medianBlur(img1,i)
e2=cv2.getTickCount()
time=(e2-e1)/cv2.getTickFrequency() #getTickFrequency()返回CPU的频率
print(time)
print(cv2.useOptimized())