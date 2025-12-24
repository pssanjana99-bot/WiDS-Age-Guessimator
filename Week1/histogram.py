import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img=cv.imread("Pictures/mountain.jpg")
cv.imshow('mountain',img)

blank=np.zeros(img.shape[:2],dtype='uint8') 
#histogram allows u to visualize the distribution of pixel intensities in an image
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale',gray)

mask=cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1)
masked=cv.bitwise_and(img,img,mask=mask)

#GRAYSCALE HISTOGRAM
gray_hist=cv.calcHist([gray],[0],None,[256],[0,256])
'''[gray] → input image (as a list)
[0] → channel index (0 = grayscale)
None → no mask (use the whole image)
[256] → number of bins in the histogram-no of bins is basicaly the interval of pixel intensities
[0, 256] → intensity range from 0 to 255'''
plt.figure()
plt.title('Grayscale img')
plt.xlabel('Bins')
plt.ylabel('The no of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()

#For masked
gray_hist=cv.calcHist([gray],[0],mask,[256],[0,256])
plt.figure()
plt.title('Grayscale img')
plt.xlabel('Bins')
plt.ylabel('The no of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()

#COLOUR HISTOGRAM
colors=('b','g','r')
for i,col in enumerate(colors):# enumerate gives index i and color value col while looping through list
    hist=cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim=([0,256])
plt.show()