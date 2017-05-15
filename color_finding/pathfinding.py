import cv2
import numpy as np


# img = cv2.imread('IMG_20170323_153127.jpg')
# img = cv2.imread('IMG_20170323_153253.jpg')
img = cv2.imread('shadowy.jpg')

img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
orig = img.copy()
cv2.imshow('original image',img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = 110
maxValue = 255
disp = np.zeros((gray.shape[0],gray.shape[1],3), dtype = 'uint8')
# Basic threshold example
th, dst = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_BINARY);
# dst = cv2.Canny(gray, 50, 100)
# Find Contours
img, contours, hierarchy = cv2.findContours(dst.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# Draw Contour
cv2.drawContours(img,contours,0,(0,255,0),0,8)
cv2.imshow("dest",dst)
# Display the resulting frame
frame = np.zeros((gray.shape[0],gray.shape[1],3), dtype = 'uint8')

for i in range(len(frame)):
    for j in range(len(frame[i])):
        # print(orig[i][j])
        if dst[i][j] != 0:
            frame[i][j] = orig[i][j] * 1
        else:
            frame[i][j] = orig[i][j] * dst[i][j]
cv2.imshow('frame',frame)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     exit
while cv2.waitKey(5) & 0xFF != ord('c'): pass
