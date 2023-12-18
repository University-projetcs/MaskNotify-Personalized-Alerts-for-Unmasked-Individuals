import cv2
import numpy as np
import os
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create()
path='Known_Faces'

def getImagewithIds(path):
    imagepaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    Ids=[]
    for imagepath in imagepaths:
        faceImg=Image.open(imagepath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        Id=int(os.path.split(imagepath)[-1].split('.')[0])
        faces.append(faceNp)
        Ids.append(Id)
        cv2.imshow('Trainning',faceNp)
        cv2.waitKey(10)
    return Ids,faces
    
Ids,Faces=getImagewithIds(path)
recognizer.train(Faces,np.array(Ids))
recognizer.save('data/Face_Training_Data.yml')
cv2.destroyAllWindows()
