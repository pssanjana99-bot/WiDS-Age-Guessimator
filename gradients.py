import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread("Pictures/mountain.jpg")
cv.imshow('mountain',img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#LAPLACIAN
lap=cv.Laplacian(gray,cv.CV_64F)
lap=np.uint8(np.absolute(lap))
cv.imshow('Laplacian',lap)

#Sobel
sobelx=cv.Sobel(gray,cv.CV_64F,1,0)
sobely=cv.Sobel(gray,cv.CV_64F,0,1)
combined=cv.bitwise_or(sobelx,sobely)
cv.imshow('Combined sobel',combined)

