from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import numpy as np
import cv2
import pyttsx3


def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
 
	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))
   
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
  
  
	return (locs, preds)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('data/Face_Training_Data.yml')
face_cascade_Path = "sources/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(face_cascade_Path)
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
Ids = []

names = ['None', 'A155','A046','A076','A116']  

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)

prototxtPath = r"sources/deploy.prototxt"
weightsPath = r"sources/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

MaskNet = load_model("data/mask_detector.model")

print("Starting video stream...")
cam = cv2.VideoCapture(0)  
vs = VideoStream(0)
cam.set(3, 250)
cam.set(4, 250)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    start_point = (15, 15)
    end_point = (370, 80)
    thickness = -1
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=400)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, MaskNet)

    for (box, pred) in zip(locs, preds):        
        (startX, startY, endX, endY) = box
        (mask, withoutmask) = pred
        
        label = "Mask" if mask > withoutmask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)      

        cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        label = "Mask" if mask > withoutmask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
 
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if(label == 'No Mask'):
                if (confidence < 100):
                    if names[id] not in Ids:
                        Ids.append(names[id])
                else:
                    id = 0
            
            print(Ids)
            cv2.putText(img, str(names[id]), (x + 5, y - 5),font, 1, (255, 255, 255), 2)

        cv2.imshow('Camera', img)
 
        if(label == 'No Mask'):
            if len(Ids)<=3:
                for i in Ids:
                    audio = "Hello"+str(i)+"please wear mask"
                    engine.say(audio)
                    engine.runAndWait()

        elif(label == 'Mask'):
            try:
                if(len(Ids)>0):
                    Ids.remove(names[id])
                                
            except ValueError as error:
                pass
            
        # else:
        #     Ids=[]
	
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(10) & 0xFF
	
    if key == 27:
        break


cv2.destroyAllWindows()
vs.stop()