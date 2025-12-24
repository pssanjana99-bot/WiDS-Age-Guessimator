import cv2 as cv
import numpy as np

img=cv.imread("Pictures/mountain.jpg")
cv.imshow('mountain',img)
cv.waitKey(0)

#TRANSLATION
def translate(img, x, y):#x and y stand for the no of pixels you want to shift
    transMat=np.float32([[1,0,x],[0,1,y]])#new_x = 1*px + 0*py + x LOGIC FOR REMEBERING THE MATRIX
    dimensions=(img.shape[1],img.shape[0])#new_y = 0*px + 1*py + y
    return cv.warpAffine(img,transMat,dimensions)
#TO REMEBER
#-X = LEFT
#-Y = UP
#X = RIGHT
#Y = DOWN

translated=translate(img,100,100)
cv.imshow('Translated',translated)

#ROTATION
def rotate(img,angle,rotPoint=None):#You can rotate img abt any point
    (height,width)=img.shape[:2]

    if rotPoint is None:
        rotPoint=(width//2,height//2)
    
    rotMat=cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions=(width,height)
    
    return cv.warpAffine(img,rotMat,dimensions)

rotated=rotate(img,45)#+45 is counter clockwise -45 is clockwise
cv.imshow('Rotated',rotated)

#RESIZING
resized=cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow('Resized',resized)

#FLIP
flip=cv.flip(img,0)
cv.imshow('Flipped',flip)
'''the second argument is flip code 
0=flip it vertically that is over the x axis
1=flip it horizontally that is over the y axis
-1=flipping the img vertically and horizontally'''
