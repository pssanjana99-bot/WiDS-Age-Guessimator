import os
import cv2 as cv
import numpy as np
p=[]
for i in os.listdir(r'C:\Users\pssan\OneDrive\Desktop\WIDS\people'):
    p.append(i)

print(p)

DIR=r'C:\Users\pssan\OneDrive\Desktop\WIDS\people'

haar=cv.CascadeClassifier('haarface.xml')

features=[]
labels=[]

def create_people():
    for person in p:
        path=os.path.join(DIR,person)
        label=p.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            faces_rect=haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_interest=gray[y:y+h,x:x+w]
                features.append(faces_interest)
                labels.append(label)
create_people()
print('Training done')
features=np.array(features,dtype='object')
labels=np.array(labels)
face_recog=cv.face.LBPHFaceRecognizer_create()
#Train the recogniser 
face_recog.train(features,labels)

face_recog.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)



