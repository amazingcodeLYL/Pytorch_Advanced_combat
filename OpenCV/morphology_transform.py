import cv2
import numpy as np
img=cv2.imread('3.jpg')
kernel=np.ones((5,5),np.uint8)
#腐蚀
# x=cv2.erode(img,kernel,iterations=1)
#膨胀
# x=cv2.dilate(img,kernel,iterations=1)
#形态学梯度
gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)

while(1):
    cv2.imshow('image',img)
    cv2.imshow('image_show',gradient)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
cv2.destroyAllWindows()