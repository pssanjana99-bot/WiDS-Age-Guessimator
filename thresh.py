import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread("Pictures/mountain.jpg")
cv.imshow('mountain',img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#THRESHOLDING-CONVERTING IMAGE TO BINARY IMAGE WHERE PIXELS R EITHER BLACK OR WHITE

#SIMPLE THRESHHOLDING
threshold'''value ex 150''',thresh'''it is thresholded image'''=cv.threshold(gray,150,255,cv.THRESH_BINARY)#if pixel intensity is above 150 it sets it to 255 if below it sets it too
cv.imshow('Simple Thresholded',thresh)
#for inverse of colour b and w
threshold,thresh_inv=cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)#if pixel intensity is above 150 it sets it to 255 if below it sets it too
cv.imshow('Simple Thresholded',thresh_inv)

#ADAPTIVE THRESHOLD-Adaptive thresholding calculates an optimum threshold value by itself without us mentioning for the image, allowing good binary conversion even when lighting is uneven or varies across the image.
adaptive=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11'''kernel size''',3'''c value is an int that is subracted frkm the mean essentialy fine tuning our mean''')
cv.imshow('Thresholding adaptive',adaptive)

#you need not only use mean thresh you can uss gaussian too it just essentily add a weight to eah pixel value and then compute mean
