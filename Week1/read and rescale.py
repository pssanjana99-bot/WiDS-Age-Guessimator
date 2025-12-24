import cv2 as cv
#FOR READING PHOTOS
img=cv.imread("Pictures/mountain.jpg")
cv.imshow('mountain',img)
cv.waitKey(0)

#FOR READING VIDEOS
capture=cv.VideoCapture(0)#Usually integer arguments refer to one or more webcam availble in the computer
capture=cv.VideoCapture('Videos/dog.mp4')
#For reading videos we use while loop and read the video frame by frame
while True:
    isTrue, frame=capture.read()
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()
#After running the video stops and an error comes specifically -215 assertion error because it ran out of frames

##TO RESCALE PHOTOS AND VIDEOS
def rescaleframe(Frame,scale=0.75):
    #will work on photos,videos,live videos
    width =int(frame.shape[1]*scale) #frame.shape[1] is width
    height=int(frame.shape[0]*scale) #frame.shape[0] is height
    dimensions=(width,height)
    
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)#This line resizes an image to the specified dimensions using area-based interpolation, which is optimal for downscaling.

##CHANGING RESOLUTION

def changeres(width,height):
    #will only work on live videos
    capture.set(3, width)
    capture.set(4, height)#capture.set(property_id, value)
