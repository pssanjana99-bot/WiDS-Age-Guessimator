import numpy as np
import os
import cv2 as cv

p=[]
for i in os.listdir(r'C:\Users\pssan\OneDrive\Desktop\WIDS\people'):
    p.append(i)

haar=cv.CascadeClassifier('haarface.xml')
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

face_recog=cv.face.LBPHFaceRecognizer_create()
face_recog.read('face_trained.yml')

img=cv.imread(r'C:\Users\pssan\OneDrive\Desktop\WIDS\people\Alyson Hannigan\2127539.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Person',gray)

#Detect the face in the image
faces_rect=haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
for (x,y,w,h) in faces_rect:
    faces_interest=gray[y:y+h,x:x+w]
    label,value=face_recog.predict(faces_interest)
    print(f'Label={p[label]} with a confidence of {value}')

    cv.putText(img,str(p[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Detected face',img)
cv.waitKey(0)