import cv2 as cv

cap=cv.VideoCapture(0)

while True:

    success,img= cap.read()
    cv.imshow('image',img)
    cv.waitKey(1)