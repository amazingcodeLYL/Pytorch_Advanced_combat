import cv2
import numpy as np
#鼠标双击的时候画圆

drawing=False
mode=True
ix,iy=-1,-1

events=[i for i in dir(cv2) if 'EVENT' in i]
# print(events)

def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        i,iy=x,y
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        if drawing==True:
            if mode==True:
                cv2.circle(img,(x,y),3,(0,0,255),-1)
    elif event==cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),-1)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False


img=np.zeros((500,500,3),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k=cv2.waitKey(1)
    if k==ord('m'):
        mode=not mode
    elif k==ord('q'):
        break


cv2.destroyAllWindows()