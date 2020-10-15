import cv2
import os
import numpy as np

kind_name=["Yangmi","Jay"]


def draw_retangle(image,rect):
    (x,y,w,h)=rect #x,y为框的左上角的点
    print(rect)
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),3)

def draw_text(image,text,x,y):
    cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2)

def detect_face(image):
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #此处转为灰度图像是因为OPenCV检测器需要灰度图像
    # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # eyes = eye_cascade.detectMultiScale(roi_gray)
    faces=face_cascade.detectMultiScale(image,scaleFactor=1.3,minNeighbors=5)
    # scaleFactor 该参数指定在每个图像比例尺上将图像尺寸缩小多少
    # minNeighbors 该参数指定每个候选矩形必须保留多少个邻居。此参数将影响检测到的面部的质量：较高的值会导致较少的检测，但质量较高
    if len(faces)==0:
        return None,None
    [x,y,w,h]=faces[0]
    return  image_gray[y:y+w,x:x+h],faces[0]

def read_training_data(train_data_path):
    labels=[]
    faces=[]
    dirs_path=os.listdir(train_data_path)
    for dir in dirs_path:
        label=int(dir)
        All_kind_path=train_data_path+"/"+dir
        Single_kind_image=os.listdir(All_kind_path)
        for sdir in Single_kind_image:
            sdir_image_path=All_kind_path+"/"+sdir
            image=cv2.imread(sdir_image_path)
            face,__=detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)

    return faces,labels

def predict(test_data):
    if test_data is None:
        return None
    test_image=test_data.copy()
    face,rect=detect_face(test_image)
    label=face_recognition.predict(face)
    kind_label=kind_name[label[0]]
    draw_retangle(test_image,rect)
    draw_text(test_image,kind_label,rect[0]-50,rect[1])
    return test_image

faces,labels=read_training_data("training_data")
#创建LBPH识别器并开始训练
face_recognition=cv2.face.LBPHFaceRecognizer_create()
face_recognition.train(faces,np.array(labels))
t1=cv2.imread("test_data/test1.jpg")
t2=cv2.imread("test_data/test2.jpg")

img1=predict(t1)
img2=predict(t2)

cv2.imshow(kind_name[0],img1)
cv2.imshow(kind_name[1],img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
