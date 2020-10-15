import cv2
import numpy as np
#常用转换为BGR↔Gray 和 BGR↔HSV
#HSV H（色彩/色度）取值范围[0,179]
# S(饱和度)取值范围[0,255] V(亮度)取值范围[0,255]
#需要注意的是不同软件取值不同，OPenCV的HSV值与其他软件HSV值对比时，记得归一化
#物体跟踪
cap=cv2.VideoCapture(0)
while(1):
    ret,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_blue=np.array([110,50,50])
    upper_blue=np.array([130,255,255])
    #根据阈值构建掩模
    mask=cv2.inRange(hsv,lower_blue,upper_blue)
    #对原图和掩模进行位运算
    res=cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
cv2.destroyAllWindows()

#