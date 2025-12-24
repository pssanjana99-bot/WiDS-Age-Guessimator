import cv2 as cv
import numpy as np

img=cv.imread("Pictures/mountain.jpg")
cv.imshow('mountain',img)

b,g,r=cv.split(img)
cv.imshow('Blue',b)
#this bascialy gives ikage in black and white and areas where it is blue it is moe whitish and other places more dark

print (img.shape)
print(b.shape)
#first gives three elemnts in a tuple but the second gives only 2 elements Splitting removes the 3rd (color) channel, so each B/G/R output becomes a 2D grayscale matrix of shape (height, width) as bgr turns into only b

#Merge
merged=cv.merge([b,g,r])
cv.imshow('Merged image',merged)

blank=np.zeros((500,500,3),dtype='uint8')
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

