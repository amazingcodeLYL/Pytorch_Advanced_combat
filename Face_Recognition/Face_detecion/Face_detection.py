import cv2
import os
# img=cv2.imread('mng.jpg')
face_classfier = 'haarcascade_frontalface_default.xml'
# eye_classfiler=''
face_cascade = cv2.CascadeClassifier(face_classfier)
# eye_cascade=cv2.CascadeClassifier(eye_classfiler)
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while(cap.isOpened()):
    ret,img=cap.read()
    # img=cv2.flip(img,-1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    # if len(faces) == 0:
    #     print("未检测到人脸")
    print("find {} faces".format(len(faces)))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0), 2)
    cv2.imshow("Image", img)
    k=cv2.waitKey(100)
    if k==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

