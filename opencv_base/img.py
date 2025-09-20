import cv2 as cv
import numpy as np

#  reading images
img = cv.imread(r"C:\Users\balam\OneDrive\Pictures\cutedp.jpg")
# cv.imshow('cutedp',img)
# cv.waitKey(0)

#  reading videos
# cap = cv.VideoCapture(r"C:\Users\balam\Downloads\catvmp4.mp4")
# while True:
#     isTrue, frame =cap.read()
#     cv.imshow('catmp4',frame)
#     if cv.waitKey(20) & 0xFF == ord('d'):
#         break
# cap.release()
# cv.destroyAllWindows()

# resize and rescale

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shappe[1]*scale)
    height = int(frame.shappe[1]*scale)
    dim =(width,height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    cap.set(3,width)
    cap.set(4,height)

# reading videos
cap = cv.VideoCapture(r"C:\Users\balam\Downloads\catvmp4.mp4")
# while True:
#     isTrue, frame =cap.read()
#     cv.imshow('catmp4',frame)
#     if cv.waitKey(20) & 0xFF == ord('d'):
#         break
# cap.release()

# drawing shapes & putting text

blank = np.zeros((600,800,3), dtype='uint8')
# cv.imshow('Blank',blank)
img = cv.imread(r"C:\Users\balam\OneDrive\Pictures\cutedp.jpg")
# cv.imshow('cutedp',img)

#1. paint the image a certain colour
blank[:] = 255,0,0 #0,0,255=red, 0,255,0=green
blank[200:300, 300:400] = 0,255,0
# cv.imshow('green',blank)
# cv.waitKey(0)

# 2. draw a rectangle
cv.rectangle(blank,(0,0),(250,250),(0,255,0), thickness=2)
cv.rectangle(blank,(0,0),(img.shape[1]//2, img.shape[0]//2), (0,0,255), thickness=cv.FILLED)
# cv.imshow('rectangle', blank)
# cv.waitKey(0)

# 3. draw a circle
cv.circle(blank,(blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255),thickness=-1)
# cv.imshow('Circle', blank)

# 4.draw a line
cv.line(blank,(100,250), (300,400), (255,255,255), thickness=3)
# cv.imshow('line', blank)
# cv.waitKey(0)

# 5.write text
cv.putText(blank,"HELLO ABI from messi! HAPPY BIRTHDAY MY MAN! BE HAPPY ALWAYS, vulem abimah", (50,400), cv.FONT_HERSHEY_TRIPLEX, 1.0,(255,0,0), 1)
# cv.imshow('Text', blank)
# cv.waitKey(0)

# blank = np.zeros((600, 800, 3), dtype="uint8")

# lines = [
#     "HELLO ABI from messi!",
#     "HAPPY BIRTHDAY MY MAN!",
#     "BE HAPPY ALWAYS, vulem abimah"
# ]

# y0, dy = 200, 50  # starting y and line spacing
# for i, line in enumerate(lines):
#     y = y0 + i*dy
#     cv.putText(blank, line, (50, y),
#                cv.FONT_HERSHEY_TRIPLEX, 1.0,
#                (255, 0, 0), 2)

# cv.imshow("Text", blank)
# cv.waitKey(0)
# cv.destroyAllWindows()


# converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)



# blur

blur=cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)
# cv.imshow('Blur',blur)


# edge cascade

canny = cv.Canny(blur,125,175)
# cv.imshow('Canny Edges',canny)

# dilated the image
di = cv.dilate(canny,(7,7),iterations=3)
# cv.imshow('Dilated',di)

# eroding
erode= cv.erode(di, (3,3),iterations=1)
# cv.imshow('Eroded',erode)

# resize
resize = cv.resize(img,(500,500), interpolation=cv.INTER_LINEAR)
# cv.imshow('Resized', resize)

# cropping

crop = img[50:200, 200:400]
# cv.imshow('Cropped', crop)

# translation
def translate (img, x,y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dim =(img.shape[1],img.shape[0])
    return cv.warpAffine(img, transMat, dim)  #-x--left, -y--up, x--right, y--down
translate=translate(img, -100,100)
# cv.imshow('Translated', translate)

# rotation
def rotate(img, angle, rp=None):
    (height,width)=img.shape[:2]
    if rp is None:
        rp =(width//2, height//2)
    rotmat= cv.getRotationMatrix2D(rp, angle,1.0)
    dimensions =(width,height)
    return cv.warpAffine(img,rotmat,dimensions)
rotated = rotate(img,35)
# cv.imshow('Rotated', rotated)

# flipping

flip = cv.flip(img,-1)
# cv.imshow('Flip', flip)
# cv.waitKey(0)

# contour detect

# ret, thresh = cv.threshold(gray, 125,125, cv.THRESH_BINARY)
# cv.imshow('Thresh', thresh)

contours, hierachies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# print(f'{len(contours)} contour(s) found!')
# cv.drawContours(blank, contours, -1,(0,0,255),1)
# cv.imshow('contours drawn', blank)

# ------------------------------------------------------------------------------


#  color spaces 
img = cv.imread(r"C:\Users\balam\Downloads\dog.jpg")
# cv.imshow('Dog', img)

# BGR to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# BGR to hsv
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow('HSV', hsv)

# BGR to lab
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# cv.imshow('LAB', lab)
cv.waitKey(0)


import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()

# BGR to rgb
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# cv.imshow('RGB', rgb)
# plt.imshow(rgb)
# plt.show()

# hsv to bgr
hsv_bgr = cv.cvtColor(img, cv.COLOR_HSV2BGR)
# cv.imshow('hsv-->bgr', hsv_bgr)
plt.imshow(hsv_bgr)
# plt.show()

# lab to bgr
lab_bgr = cv.cvtColor(img, cv.COLOR_Lab2BGR)
# cv.imshow('LB', lab_bgr)
plt.imshow(lab_bgr)
# plt.show()

# channels

b,g,r =cv.split(img)
blank = np.zeros(img.shape[:2], dtype="uint8")
bl= cv.merge([b,blank,blank])
gr= cv.merge([blank,g,blank])
re= cv.merge([blank,blank,r])

# cv.imshow('Blue',bl)
# cv.imshow('Green',gr)
# cv.imshow('Red',re)
# print(img.shape)
# print(b.shape)
# print(g.shape)
# print(r.shape)
merged = cv.merge([b,g,r])
# cv.imshow("Merged", merged)


# BLUR technique
# averaging

avg = cv.blur(img,(8,8))
# cv.imshow('average blur', avg)

# gaussian blur
gauss = cv.GaussianBlur(img, (7,7),0)
# cv.imshow('Gauss_blur', gauss)

# median blur
median = cv.medianBlur(img, 7)
# cv.imshow('Median Blur', median)

# bilateral blur
bil = cv.bilateralFilter(img,5,15,15)
# cv.imshow('Bilateral', bil)


# bitwise operator
blank = np.zeros((400,400), dtype='uint8')
rect = cv.rectangle(blank.copy(),(30,30),(370,370),255, -1)
circle = cv.circle(blank.copy(), (200,200),200,255,-1)
# cv.imshow('Rectangle',rect)
# cv.imshow('Circle',circle)

# bitwise and --> intersecting regions
ba= cv.bitwise_and(rect,circle)
# cv.imshow('using AND',ba)

# bitwise or -->non-intersecting regions
ba= cv.bitwise_or(rect,circle)
# cv.imshow('using OR',ba)

# bitwise xor
ba= cv.bitwise_xor(rect,circle)
# cv.imshow('using XOR',ba)

# bitwise Not
ba= cv.bitwise_not(circle)
# cv.imshow('using NOT',ba)


blank=np.zeros(img.shape[:2], dtype='uint8')
# cv.imshow('Blank Img.',blank)

# masking 
mask = cv.circle(blank,(img.shape[1]//2, img.shape[0]//2), 100,255,-1)
# cv.imshow('Mask',mask)

masked = cv.bitwise_and(img,img, mask=mask)
# cv.imshow('Masked Image',masked)

wired_shape=cv.bitwise_and(circle,rect)
# cv.imshow('Wired Shape', wired_shape)
# masked = cv.bitwise_and(img,img, mask=wired_shape)
# cv.imshow('Wired Shape', masked)

# histogram computation
gray_hist=cv.calcHist([gray],[0],None, [256],[0,256])
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
# plt.show()

#  colour hist
colors=('b','g','r')
for i,col in enumerate(colors):
    hist=cv.calcHist([img],[i],mask,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])
# plt.show()

# Thresholding
# simple threshold

threshold, thresh=cv.threshold(gray,225,255,cv.THRESH_BINARY)
# cv.imshow('simple thresh', thresh)

threshold, thresh_inv=cv.threshold(gray,225,255,cv.THRESH_BINARY_INV)
# cv.imshow('simple thresh', thresh_inv)

# adaptive thresh
at=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 8)
# cv.imshow('Adaptive thresh', at)


cv.waitKey(0)



       



