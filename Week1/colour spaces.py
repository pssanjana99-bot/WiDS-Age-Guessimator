import cv2 as cv
import numpy as np

img=cv.imread("Pictures/mountain.jpg")
cv.imshow('mountain',img)
#COLOUR SPACES ARE BASICALLY SPACE OF COLOURS,A SYSTEM REPRESNTING AN ARRAY OF PIXEL COLOURS(RGB IS COLOURSPACE GREYSCALE IS COLOUR SPACE)

#BGR TO GREYSCALE
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Grey',gray)

#BGR TO HSV
hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
cv.imshow('Hsv',hsv)

#BGR TO LAB
lab=cv.cvtColor(img,cv.COLOR_BGR2Lab)
cv.imshow('LAB',lab)

#BGR TO RGB
rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)

#Make sure to keep in mind that open cv has bgr format but every other lib has rgb and for ex if you tak this image to matplotlib and pplot invesrion of oclur takes place as it has no idea opencv has brg colour so make sure to convert4


cv.waitKey(0)