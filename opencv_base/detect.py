import cv2 as cv
import numpy as np

#  reading images
img = cv.imread(r"C:\Users\balam\Downloads\dog.jpg")
# cv.imshow('cutedp',img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('dp', gray)

# laplacian
lap=cv.Laplacian(gray,cv.CV_64F)
lap=np.uint8(np.absolute(lap))
# cv.imshow('Laplacian', lap)

# sobel
sobelx=cv.Sobel(gray,cv.CV_64F,1,0)
sobely=cv.Sobel(gray,cv.CV_64F,0,1)
combine_model = cv.bitwise_or(sobelx,sobely)
# cv.imshow('sobelx',sobelx)
# cv.imshow('sobely',sobely)
# cv.imshow('combined model',combine_model)
canny = cv.Canny(gray, 150,175)
# cv.imshow('canny img', canny)

cv.waitKey(0)
