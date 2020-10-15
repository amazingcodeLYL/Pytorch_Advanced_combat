import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('1.jpg')
print(img.shape)
print(img.dtype)
px=img[100,100] #通过坐标获取该点RGB像素值
print(px)
blue=img[100,100,0]
img[101,101]=[255,255,255]
print(img[101,101])
print(img.item(10,10,2))#单个获取RGB中的值
img.itemset((10,10,2),100)
print(img.item(10,10,2))


r,g,b=cv2.split(img)#拆分RGB
cv2.imshow("image",r)
img=cv2.merge([r,g,b])
b=img[:,:,0]
img[:,:,2]=0
reflect=cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REFLECT)

plt.subplot(231),plt.imshow(reflect,'gray'),plt.title('replicate')
# plt.show()

# cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()