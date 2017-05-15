import cv2
import cvk2
import numpy as np
import ros

def within_limits(theta):
    theta[theta < 50] = 0
    theta[theta > 0] = 1

def isBlue(img):
    isBlue = np.zeros((img.shape[0],img.shape[1],3), dtype = 'uint8')
    b_matrix, g_matrix, r_matrix = cv2.split(img)
    theta_b_g = np.arctan2(b_matrix,g_matrix) * 180/np.pi
    theta_b_r = np.arctan2(b_matrix,r_matrix) * 180/np.pi
    within_limits(theta_b_g)
    within_limits(theta_b_r)
    blue = theta_b_g*theta_b_r
    blue = blue.astype('uint8')
    return blue

def vision():
    cap = cv2.VideoCapture("test_data/vid.mp4")
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.pyrDown(frame)
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = 60
        maxValue = 255
        # disp = np.zeros((gray.shape[0],gray.shape[1],3), dtype = 'uint8')
        # # Basic threshold example
        th, dst = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_BINARY);
        res = cv2.bitwise_and(frame,frame,mask = dst)
        # cv2.imshow("masked",res)
        blu = isBlue(res)
        blue = cv2.bitwise_and(frame,frame,mask = blu)
        blu[blu > 0] = 255
        cv2.imshow("blue",blue)
        cv2.imshow("orig",blu)
        # dst = cv2.Canny(blue, 50, 100)
        # cv2.imshow("canny",dst)
        # # Find Contours
        print(blu.shape)
        img, contours, hierarchy = cv2.findContours(blu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # # Draw Contour
        for i in  range(len(contours)):
            info = cvk2.getcontourinfo(contours[i])
            if info['area'] > 100:
                cv2.drawContours(frame,contours,i,(255,0,0),-1)
        cv2.imshow("contours",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    vision()
