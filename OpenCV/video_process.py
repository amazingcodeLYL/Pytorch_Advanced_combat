import numpy as np
import cv2
cap=cv2.VideoCapture(0) #0默认为计算机默认摄像头 1可以更换来源

fourcc=cv2.VideoWriter_fourcc(*'XVID')
#视频编码风格有DIVX , XVID , MJPG , X264 , WMV1 , WMV2
#XVID是最好的，MJPG是高尺寸视频，X264得到小尺寸视频
out=cv2.VideoWriter('output.avi',fourcc,20.0,(640,480)) #第一个参数是文件输出名，指定FourCC编码 ，帧/秒，窗口大小

while(cap.isOpened()):
    ret,frame=cap.read()
    #ret->True，False 表示有没有读取到图片
    #frame表示截取到一帧得图片
    if ret==True:
        # frame=cv2.flip(frame)
        #cv2.flip表示图像翻转 后面参数1是水平翻转 0垂直翻转 -1水平垂直翻转
        out.write(frame)
    # gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #cv2.cvtColor颜色空间转换
        #opencv有多种多彩空间，包括RGB，HSI，HSV，HSB，YCrCb，CIE XYZ、CIE Lab8种
        #用函数将图像从RGB转为HSV，对应函数是BGR2HSV,在opencv中默认颜色空间是BGR
        cv2.imshow('frame',frame)
        if cv2.waitKey(1)&0xFF==ord('q'):#按q键退出
            break
cap.release()
out.release()
cv2.destroyAllWindows()
