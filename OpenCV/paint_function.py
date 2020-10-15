import numpy as np
import cv2
img=np.zeros((512,512,3),np.uint8)
# cv2.line(img,(0,0),(260,260),(255,0,0),5) #画直线
#cv.Line(img, pt1, pt2, color, thickness=1, lineType=8, shift=0) plt1和plt2是直线的起点和终点
#画矩形
# cv2.rectangle(img,(350,0),(500,180),(0,0,255),3)
#画圆
# cv2.circle(img,(425,63),63,(0,0,255),-1) #-1表示向内填充
# 画椭圆
# cv2.ellipse(img,(256,256),(100,50),0,0,360,255,-1)

#画多边形
# pts=np.array([[10,5],[20,30],[70,20]],np.int32)
# pts=pts.reshape((-1,1,2))
# cv2.polylines(img,[pts],True,(0,255,255))
#########################


#在图片上添加文字
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,300),font,4,(255,255,255),2,cv2.LINE_AA)

cv2.namedWindow('image',cv2.WINDOW_NORMAL) #新建一个窗口
cv2.resizeWindow('image',1000,1000)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

