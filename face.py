import numpy as np 
import cv2
import pickle
import math

face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

#recognizer
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

#forname
labels={"person_name":1}
with open("labels.pickle","rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

#Capture Video
cap=cv2.VideoCapture(0)

while(True):
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
       

        #Recognize?
        id_,conf=recognizer.predict(roi_gray)
        if conf>=45:
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            t=name+": "+str(math.ceil(conf))
            color=(0,0,255)
            stroke=2
            cv2.putText(frame,t,(x,y),font,1,color,stroke,cv2.LINE_AA)
        img_item="my-image.png"
        cv2.imwrite(img_item,roi_gray)





        #Rectangle
        color=(0,255,0)#BGR 0-255
        stroke=2
        width=x+w
        height=y+h
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)


    cv2.imshow('Frame',frame)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()