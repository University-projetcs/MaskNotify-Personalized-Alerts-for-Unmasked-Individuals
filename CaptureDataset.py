import cv2
import numpy as np

cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
count=0

faceCascade=cv2.CascadeClassifier('sources/haarcascade_frontalface_default.xml')

face_id=input('\n enter your rollno. and press enter ===> ')
print('\n [INFO] Initializeing face capture. Please look at the camera :)')

while True:
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        count+=1
        cv2.imwrite('Known_Faces/'+str(face_id)+'.'+str(count)+'.jpg',gray[y:y+h,x:x+w])
        cv2.imshow('capturing...',img)
        
    k=cv2.waitKey(100) & 0xff
    if k<30:
        break
    elif count>=50:
        break
    
print('\n [INFO] Data collected ✓')


