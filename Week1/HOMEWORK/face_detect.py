import cv2 as cv

img=cv.imread('Pictures/lady.jpg')
cv.imshow('Person',img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

haar=cv.CascadeClassifier('haarface.xml')
faces_rect=haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)#Runs Haar Cascade face detection on the grayscale image, scaling the image by 1.1 each pass and requiring at least 3 neighbor detections to confirm a face.
print(f'Number of faces found ={len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
'''Loops through every detected face and draws a green rectangle
 from the top-left corner (x, y) to the bottom-right corner (x+w, y+h) with thickness 2.'''
cv.imshow('Detected faces',img)

#haarcascades r really sensitive to noise and anythign tha tlooks like a face it detects

cv.waitKey(0)
