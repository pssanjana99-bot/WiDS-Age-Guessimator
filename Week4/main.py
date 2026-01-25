import numpy as np
import cv2 as cv

FACE_PROTO="models/deploy.prototxt"
FACE_MODEL="models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
AGE_PROTO="models/age_deploy.prototxt"
AGE_MODEL="models/age_net.caffemodel"
GENDER_PROTO="models/gender_deploy.prototxt"
GENDER_MODEL="models/gender_net.caffemodel"

AGE_BLOCKS=['0-2','4-6', '8-12','15-20', '25-32','38-43','48-53', '60+']
GENDER_BLOCKS = ['Male', 'Female']
CONFI_LIMIT=0.7
MEAN_VALUES = (104, 117, 123)


def load():
    face_net=cv.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net=cv.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net=cv.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    return face_net, age_net, gender_net

def read_resize(path):
    img=cv.imread(path)
    return cv.resize(img,(720,640))

def find(net,frame):
    h,w=frame.shape[:2]
    blob=cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123), swapRB=False, crop=False)
    net.setInput(blob)
    detections=net.forward()

    faces = []

    for i in range(detections.shape[2]):
        conf=detections[0,0,i,2]
        if conf>CONFI_LIMIT:
            box=(detections[0,0,i,3:7] * np.array([w,h,w,h])).astype(int)
            faces.append(box)
    return faces

def analyze(face_img, age_net, gender_net):
    blob=cv.dnn.blobFromImage(face_img, 1.0, (227,227), MEAN_VALUES)
    gender_net.setInput(blob)
    gender = GENDER_BLOCKS[gender_net.forward().argmax()]
    age_net.setInput(blob)
    age = AGE_BLOCKS[age_net.forward().argmax()]
    age_probs = age_net.forward()[0]#This code stabilizes age prediction by averaging the two most probable age groups instead of blindly choosing the top one for better accuracy
    best = np.argsort(age_probs)[-2:]
    age = AGE_BLOCKS[int(np.mean(best))]
    return gender,age

def display(frame, faces, age_net, gender_net):
    for (x1,y1,x2,y2) in faces:
        face=frame[y1:y2, x1:x2]
        gender,age = analyze(face,age_net,gender_net)
        text = f"{gender} | {age}"
        cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv.putText(frame,text,(x1,y1-10),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
    return frame

face_net, age_net, gender_net=load()
img=read_resize("img_1.jpg")
faces = find(face_net, img)
final = display(img, faces,age_net, gender_net)
cv.imshow("Age & Gender Detector",final)
cv.waitKey(0)
cv.destroyAllWindows()

