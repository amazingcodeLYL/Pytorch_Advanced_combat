import cv2
import numpy
from matplotlib import  pyplot as plt
img=cv2.imread('1.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow("Image",img)
k=cv2.waitKey (0)
if k==27:
    cv2.destroyAllWindows()
elif k==ord('s'):
    cv2.imwrite('46.png',img)
    cv2.destroyAllWindows()

plt.imshow(img,cmap='gray',interpolation='bicubic')
plt.xticks([]),plt.yticks([])
plt.show()
