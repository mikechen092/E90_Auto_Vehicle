#!/usr/bin/env python
import cv2
import cvk2
import numpy as np
import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cPickle as pickle
import os

#pickle is where we store our classifier
input_pickle = open("src/control/scripts/picklejar.pkl","rb")
poly_svm = pickle.load(input_pickle)
input_pickle.close()
cap = cv2.imread('')
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("video1.avi",fourcc,20,(1280,480))

#Makes polynomial from liner SVC
def make_input(data):
    assert data.shape[1] == 3
    # make data into 0's/1's
    data = data.astype(np.float32) / 255.0

    # columns 0 1 2 0*0 0*1 0*2 1*1 1*2 2*2
    d0 = data[:,0].reshape(-1,1)
    d1 = data[:,1].reshape(-1,1)
    d2 = data[:,2].reshape(-1,1)

    #return data
    result = np.hstack((data, d0*d0, d0*d1, d0*d2, d1*d1, d1*d2, d2*d2))

    return result


#
def vision():

    while(True):

        ret, frame2 = cap.read()

        shape = frame2.shape


        frame = cv2.resize(frame2,(320,240),interpolation = cv2.INTER_CUBIC)

        result = poly_svm.predict(make_input(frame.reshape(-1,3)))

        img = result.reshape(frame.shape[:2]) * 255

        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        print "img_color",img_color.shape
        print "frame",frame.shape

        kernel = np.ones((5,5),np.uint8)
        kernel2 = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
        erosion = cv2.erode(closing,kernel2,iterations = 10)
        size = erosion.shape


        black = np.zeros((size[0]/4,size[1],3), np.uint8)
        black2 = np.zeros((size[0]/4,size[1]), np.uint8)
        erosion = erosion[(size[0]/4):3*size[0]/4,:]

        erosion = np.vstack((black2,erosion,black2))
        img_color = img_color[(size[0]/4):size[0]*3/4,:]

        img_color = np.vstack((black,img_color,black))

        frame = frame[(size[0]/4):size[0]*3/4,:]
        frame = np.vstack((black,frame,black))

        img, contours,hierarchy = cv2.findContours(erosion, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        cx = None
        cy = None
        ledge = None
        redge = None

        try:
            areas = [cv2.contourArea(contour) for contour in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            x,y,w,h = cv2.boundingRect(cnt)
            ledge = x
            redge = x+w
            cv2.rectangle(img_color,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.drawContours(img_color, contours, max_index, (255,255,0),-1)
            cv2.circle(img_color,(cx,cy),5,(0,255,0),5)
        except:
            pass

        data = (cx,cy,frame.shape[1],ledge,redge)
        img_color = cv2.resize(img_color,(shape[1],shape[0]),interpolation = cv2.INTER_CUBIC)
        combined = np.hstack((frame2,img_color))
        writer.write(combined)
        print("combined",combined.shape)
        return data, combined

def talker():
    cv_bridge = CvBridge()
    img_topic = rospy.Publisher('image_topic2',Image,queue_size=10)
    pub = rospy.Publisher('path_finding',numpy_msg(Floats),queue_size=10)
    rospy.init_node('talker',anonymous=True)
    rate = rospy.Rate(.01)
    while not rospy.is_shutdown():
        data,img = vision()
        a = np.array([data[0],data[1],data[2],data[3],data[4]],dtype = np.float32)
        rospy.loginfo(a)
        pub.publish(a)
        img_topic.publish(cv_bridge.cv2_to_imgmsg(img, "bgr8"))

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    talker()
