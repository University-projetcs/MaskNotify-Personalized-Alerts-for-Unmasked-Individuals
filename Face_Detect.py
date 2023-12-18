import cv2
import numpy as np
import os

cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('data/Face_Training_Data.yml')
cascadePath = "sources/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
names=['','A155','A046','A076','A116']

id = 0
Ids=[]

cam.set(3, 250) 
cam.set(4, 250) 
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
     ret, img =cam.read()
     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     
     faces = faceCascade.detectMultiScale( 
          gray,
          scaleFactor = 1.2,
          minNeighbors = 5,
          minSize = (int(minW), int(minH)),
     )
     
     for(x,y,w,h) in faces:
          cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
          id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
          
          if (confidence < 100):
               confidence = "  {0}%".format(round(100 - confidence))
          else:
               confidence = "  {0}%".format(round(100 - confidence))
          
          box_coords = ((x, y), (x + w + 2,y-30))
          cv2.rectangle(img, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
          cv2.putText(img,str(names[id]),(x+5,y-5),font,1,(255,255,255),2)
          cv2.putText(img,str(confidence),(x+5,y+h-5),font,1,(255,255,0),1)  
          
          if not names[id] in Ids:
               Ids.append(names[id])

     cv2.imshow('Face',img) 
     k=cv2.waitKey(10) & 0xff
     if k == 27:
          break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

