import cv2 as cv
import numpy as np

img=cv.imread("Pictures/mountain.jpg")
cv.imshow('mountain',img)

blank=np.zeros((400,400),dtype='uint8')

rectangle=cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)#-1 for filling this img
circle=cv.circle(blank.copy(),(200,200),200,255,-1)

#bitwise AND- takes two images place them on top of each other and return the intersection so the intersection regionis white and the rest are black
bitwise_and=cv.bitwise_and(rectangle,circle)

#bitwise OR-takes two images place them on top of each other and return intersecting and non t=intersecting region all basically the whole shape you get
bitwise_or=cv.bitwise_or(rectangle,circle)

#bitwise XOR-returns non intersecting regions
bitwise_xor=cv.bitwise_xor(rectangle,circle)

#bitwise NOT-it inverts the binary colour
bitwise_not=cv.bitwise_not(rectangle)