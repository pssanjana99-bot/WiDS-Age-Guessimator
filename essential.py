import cv2 as cv
img=cv.imread("Pictures/mountain.jpg")
cv.imshow('mountain',img)

#converting image to greyscale
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Grey',gray)

#Blur an image
blue=cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT) #The kernel number should be odd like ex 3,3 and big kernels means more blur
cv.imshow('Blur',blur)

#EDGE CASCADE
canny=cv.Canny(img,125,175) # this means if intensity of border>175 keep if btw 125 and 175 see if its associated wiht a border and keep it and intensity<125 ignore

#Dilating the image
dilate=cv.dilate(canny,(3,3),iterations=1)#makes the edges thicker
cv.imshow('Dilated',dilate)

#Eroding
erode=cv.erode(Dilated,(3,3),iterations=1)#Iteration is basically how many times dilate/erode is applied for increasijg thickness
cv.imshow('Eroded',erode)
#Eroding is to get back the normla image fromm the dilated image

#RESIZE 
resize=cv.resize(img,(500,500),interpolation=cv.INTER_AREA) #It resizes the org img to 500,500 without changing aspect
cv.imshow('Resize',resize)

#note: for increasing image size use cv.INTER_LINEAR 
#for decreasing img size use cv.INTER_AREA
#cv.INTER_CUBIC is also used for increasing img size but it is slower but gives image of better quality

#CROPPING
crop=img[50:200,200:400]
cv.imshow('cropped',crop)



cv.waitKey(0)