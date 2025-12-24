import cv2 as cv
import numpy as np

img=cv.imread("Pictures/mountain.jpg")
cv.imshow('mountain',img)


#masking=you can focus on specific parts of img which you want to focus if you want to focus on some ppl in the img you can mask over the people and remove all the unwanted parts of the img
#condtion for masking both the blank and the  img has to be of the same size
blank=np.zeros(img.shape[:2],dtype='uint8') 

mask=cv.circle(blanl,(img.shape[1]//2,img.shape[0]//2),100,255,-1)
masked=cv.bitwise_and(img,img,mask=mask)
cv.imshow('Masked',masked) #displays img of cat with only the cat face in the centre that i of the circle area rest all black
