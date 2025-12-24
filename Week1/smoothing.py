import cv2 as cv
import numpy as np

img=cv.imread("Pictures/mountain.jpg")
cv.imshow('mountain',img)

#AVERGAING-Each pixel is replaced by the average value of the pixels around it.\
'''so basically what it does is open a square window and its centre pixel density is calulated by average of all the pixels surrounding it and it goes on covering all the pixels inn the image'''
average=cv.blur(img,(3,3))#more kernel size more blur
cv.imshow('Averaged',average)

#GAUSSIAN BLUR-Gaussian Blur smooths the image by averaging pixels using a weighted Gaussian kernel, which reduces noise but preserves edges better than normal averaging.
blur=cv.GaussianBlur(img,(3,3),0) #If sigma is set to 0, OpenCV automatically calculates the sigma from the kernel size sigma take into accound how spread out the blur is 
cv.imshow('Blur',blur)

'''sigma can be though of like if the square window is in top left corner a pixel from bottom right corner affecting the centre pixel in averagin formulae is what sigma signifies
'''

#MEDIAN BLUR-almost same as avergaing  just that instead of dning average of the pixels it finds the median of the pixels
median=cv.medianBlur(img,3)#just 3 is enough for this as opencv assumes it is 3,3
cv.imshow('Median',median)

#Bilateral-most effective blurring- you get blurred images but edges r retained
bilateral=cv.bilateralFilter(img,5'''note that it is not kernel size here but diameter''',15'''larger value means more colours r considered when the blur is computed''',15'''this is the sigma''')
cv.imshow('Bilateral',bilateral)




