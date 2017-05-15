import cv2
import cvk2
import numpy as np
import math
from matplotlib import pyplot as plt

def within_limits(theta):
    theta[theta < 40] = 0.0
    theta[theta > 53] = 0
    theta[theta > 0] = 1

def isGray(img):
    print img.dtype
    isGray = np.zeros((img.shape[0],img.shape[1],3), dtype = 'uint8')
    b_matrix, g_matrix, r_matrix = cv2.split(img)
    theta_r_g = np.arctan2(r_matrix,g_matrix) * 180/np.pi
    theta_b_g = np.arctan2(b_matrix,g_matrix) * 180/np.pi
    theta_r_b = np.arctan2(b_matrix,r_matrix) * 180/np.pi
    within_limits(theta_r_g)
    within_limits(theta_b_g)
    within_limits(theta_r_b)
    gray = theta_r_g*theta_b_g*theta_r_b
    b = (b_matrix*gray).astype('uint8')
    g = (g_matrix*gray).astype('uint8')
    r = (r_matrix*gray).astype('uint8')

    isGray = cv2.merge((b,g,r))
    return isGray

def thresholding(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # feel free to mess with these values!!!! just let us know which values worked the best for you (try to keep them consistant across all data sets tho)
    ret, thresh = cv2.threshold(imgray, 130, 200, cv2.THRESH_BINARY)
    # list path is going to be every pixel that is non zero in the image which is what we define as the "path"
    kernel = np.ones((10,10), np.uint8)
    # cv2.imshow('unmorphed',thresh)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('thresholded image',closing)
    for i in range(len(img)):
        for j in range(len(img[i])):
            if i < (len(img)/4):
                img[i][j] *= 0
                closing[i][j] *= 0
            if closing[i][j] == 0:
                img[i][j] *= 0
    return closing

def lines(closing, orig):
    edges = cv2.Canny(closing,130,200,3)
    lines = cv2.HoughLines(edges,1,np.pi/180,50)
    # cv2.imshow('canny',edges)

    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(orig,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow('line', orig)

def main():
    cap = cv2.VideoCapture("test_data/vid.mp4")
    while(True):
        ret, img = cap.read()
        # img = cv2.imread("test_data/IMG_20170323_153118.jpg")
        # img = cv2.pyrDown(img)
        # img = cv2.pyrDown(img)
        orig = img.copy()
        # feel free to mess with how much blur there is in the image
        img = cv2.medianBlur(img,5)
        gray = isGray(img)
        cv2.imshow('isGray',gray)
        closing = thresholding(gray)
        lines(closing, orig)
        cv2.imshow("win",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # while cv2.waitKey(5) & 0xFF != ord('c'): pass
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
