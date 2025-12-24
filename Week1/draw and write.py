import cv2 as cv
import numpy as np
blank=np.zeros((500,500,3),dtype='uint8')#uint8 is image datatype and we r giving to height width and number of colour channels wich is equal to 3
cv.imshow('Blank',blank)
#PAINT THE IMG A CERTAIN COLOUR
blank[:]=0,255,0 #[:] is to reference all the pixels and 0,255,0 comes in reference to bgr where no blue full green no red
cv.imshow('Green',blank)
#You can also colour only certain pixels by giving it specific range like
blank[200:300,300:400]

#TO DRAW A RECTANGLE
cv.rectangle(blank,(0,0),(250,250),(0,250,0),thickness=2)
cv.imshow('Rectangle',blank)

#if you want to fill the rectangle wiht colour then
cv.rectangle(blank,(0,0),(250,250),(0,250,0),thickness=cv.FILLED) #OR
cv.rectangle(blank,(0,0),(250,250),(0,250,0),thickness=-1)

#you can alwso mention (250,250) differently like
cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,250,0),thickness=cv.FILLED)

#DRAW A CIRCLE
cv.circle(blank,(250,250),40,(0,0,255),thickness=3)
cv.imshow('circle',blank)

#DRAW A LINE
cv.line(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,250,0),thickness=3)#draws a line from 0,0 to 250,250

#HOW TO WRITETEXT ON AN IMAGE
cv.putText(blank,'Hello',(225,225),cv.FONT_HERSHEY_TRIPLEX,1.0,,(0,255,0),thickness=2)

cv.waitKey(0)

