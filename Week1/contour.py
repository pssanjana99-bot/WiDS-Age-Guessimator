import cv2 as cv
import numpy as np

img=cv.imread("Pictures/mountain.jpg")
cv.imshow('mountain',img)

#CONTOURS ARE BASCIALLY THE BOUNDARIES OF OBJECTS , THE LINE OR CURVE THAT JOIN THE CONTINUOS POINTS ALONG THE BOUNDARY OF AN OBJECT
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Grey',gray)

blank=np.zeros((500,500,3),dtype='uint8')#uint8 is image datatype and we r giving to height width and number of colour channels wich is equal to 3
cv.imshow('Blank',blank)

canny=cv.Canny(img,125,175)
cv.imshow('Cannied',canny)
#OR
ret,thresh=cv.threshold(gray,125,255,cv.THRESH_BINARY)#tries tio binarise if the density of the pixel is below 125 it is set to 0 or black and if its above it is set to whitee


contours,hierarchies=cv.findContours(canny'''OR thresh''',cv.RETR_LIST,cv.CHAIN_APPROX_NONE)#retrtree if you want only the heirarchial contours retr external if you want only the external contours and retr list if you want all the contours
'''contours bascially is all the coordiantes where cotnours r found , it is a list
hierarchis are bascally out of scope for this course but like if there is rectangl inside that sq inside that circle it basicaly represnts this hierarchy
cv.chain approx basicalyl is how we want to appro the contour 
none does nothign jsut returs all the contours
simple compresses all the contours into simple ones that makes more sense ex if none gives all the points btw a line simple only takes the two end points because it makes more sense
'''
print(f'{len(contours)} contour(s) found!')

#draw contours method
cv.drawContours(blank,contours,-1'''-1 means show all the contours''',(0,0,255),1'''ths is thickness''')
cv.imshow('Contours',blank)

#ITS BETTTER TO USE CANNY AND THEN FIND CONTOURS INSTEAD OF THRESHHOLDING AND FINDING COTNOURS AS IT HAS ITS OWN DISADVANTAGES

cv.waitKey(0)