import cv2
import numpy as np

cap = cv2.VideoCapture('vid.mp4')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = 110
    maxValue = 255
    disp = np.zeros((gray.shape[0],gray.shape[1],3), dtype = 'uint8')
    # Basic threshold example
    th, dst = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_BINARY);
    # dst = cv2.Canny(gray, 50, 100)
    # Find Contours
    img, contours, hierarchy = cv2.findContours(dst.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    # Draw Contour
    cv2.drawContours(frame,contours,0,(0,255,0),0,8)
    cv2.imshow("dest",dst)
    # Display the resulting frame
    # cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
