import cv2
import numpy as np
def nothing(x):
    pass

img=np.zeros((300,512,3),np.uint8)
cv2.namedWindow('image')

cv2.createTrackbar('R','image',0,255,nothing)
#参数依次是滑条名称，窗口名称，最小值，最大值，滑条每次被拖动时回调的函数
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)

switch='0:OFF\n1:ON'
cv2.createTrackbar(switch,'image',0,1,nothing)
while(1):
    cv2.imshow('image',img)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
    r=cv2.getTrackbarPos('R','image') #得到滑条的位置
    g=cv2.getTrackbarPos('G','image')
    b=cv2.getTrackbarPos('B','image')
    s=cv2.getTrackbarPos(switch,'image')

    if s==0:
        img[:]=0
    else:
        img[:]=[r,g,b]

cv2.destroyAllWindows()