#!/usr/bin/env python
import rospy
import roslib
import time
import serial
import con
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import numpy

# control updates 100Hz or every .01 seconds
CONTROL_PERIOD = rospy.Duration(.001)

# base forward speed when following blobs
BASE_SPEED = 1000

class Controller:

    def __init__(self):

        #Usb port used to interface with kangaroo motor controller
        port = "/dev/ttyACM0"
        baud = 9600

        self.ser = serial.Serial(port, baud, timeout=1)

        if self.ser.isOpen():
            print self.ser.name + 'is open...'
        else:
            raise("SERIAL NOT FOUND")
        self.joy = con.hci_init()

        #initialize a node to control the motors
        rospy.init_node('controller')

        self.r_prev = 0

        self.l_prev = 0

        self.r_speed = 0

        self.l_speed = 0

        self.position = None

        self.screen_size = None

        self.ledge = None

        self.redge = None
        # need to subscribe to the topic that outputs vision data
        rospy.Subscriber('path_finding',numpy_msg(Floats),self.vision_callback)

        # need to set up how often we need to call control_callback
        rospy.Timer(CONTROL_PERIOD,self.control_callback)

        # create a variable to tell if dead man switch is active
        self.should_stop = True

        # initialize the state to stop
        self.state = "stopped"

        #Initialize 2 motors
        # self.ser.write("1,start")
        # self.ser.write("2,start")


    #Writes motor speeds to kangaroo
    def do_motion(self):
        rospy.loginfo(self.state)
        tol = 10

        # if self.state == "following":
        #Cap on change in velocity, smoother ride
        if self.r_speed - self.r_prev > tol:
            self.r_speed = self.r_prev + tol
        if self.r_speed - self.r_prev < -tol:
            self.r_speed = self.r_prev - tol
        if self.l_speed - self.l_prev > tol:
            self.l_speed = self.l_prev + tol
        if self.l_speed - self.l_prev < -tol:
            self.l_speed = self.l_prev - tol

        #Writes speed to kangaroo
        rospy.loginfo("M1:"+str(int(self.r_speed)))
        rospy.loginfo("M2:"+str(int(self.l_speed)))
        self.ser.write("M1:"+str(-int(self.r_speed))+'\r\n')
        self.ser.write("M2:"+str(int(self.l_speed))+'\r\n')

        #sets new previous speed
        self.l_prev = self.l_speed
        self.r_prev = self.r_speed

    # converts angular speed into a delta to change the wheel speeds
    def convert(self):
        # gain value
        K = 1.5

        # calculates screen position from screen size /2 - the position of the blob
        s_position = (self.screen_size/2) - self.position

        # find the relative position of the blob in terms of the screen 1 for far left -1 for far right 0 for center
        r_position = (s_position)/(self.screen_size/2)

        # calculates the delta fromt the relative position and the base speed multiplied by the gain value
        delta = r_position * BASE_SPEED * K

        return delta


    # vision callback does stuff with vision data from topic
    def vision_callback(self,data):
        # first item in data should be the position of the largest contour
        if numpy.isnan(data.data[0]):
            self.position = None
        else:
            self.position = (data.data[0])
        rospy.loginfo(self.position)
        #
        # # second item in data should be the screen size of the camera
        self.screen_size = (data.data[2])
        self.ledge = data.data[3]
        self.redge = data.data[4]

    # control callback does stuff every so often set with CONTROL_PERIOD
    def control_callback(self, timer_event=None):
        # if the state is stopped
        if self.state == "stopped":
            # set the wheel velocities to zero
            self.r_speed = 0
            self.l_speed = 0

            # if the deadman switch is pressed then check if there are any blobs
            if con.hci_button(self.joy,5):

                # if there are no blobs then transition into ready state to search for blobs
                if self.position == None:
                    self.state = "ready"

                # if there are blobs then transition into follow state to follow blobs
                else:
                    self.state = "following"

        elif self.state == "ready":
            self.r_speed = 0
            self.l_speed = 0
            # if the deadman switch is released then go to stopped state
            if con.hci_button(self.joy,5) == False:
                self.state = "stopped"

            # deadman still active
            else:
                # if there are blobs then go to follow state
                if self.position != None:
                    self.state = "following"

        # if the deadman is active and there are blobs
        elif self.state == "following":
            # if there are no blobs to be seen change states
            if self.position == None:
                # check if deadman is active transition appropriately
                if con.hci_button(self.joy,5):
                    self.state = "ready"
                else:
                    self.state = "stopped"

            # if you see a blob
            else:
                # make sure dead man is active else transition to the stopped state
                if con.hci_button(self.joy,5) == False:
                    self.state = "stopped"

                # is the deadman is active the do move
                else:
                    # calculates how much each wheel speed must change
                    delta = self.convert()

                    self.r_speed = BASE_SPEED - delta
                    if self.r_speed > 2047:
                        self.r_speed = 2047
                    self.l_speed = BASE_SPEED + delta
                    if self.l_speed > 2047:
                        self.l_speed = 2047

        # send the velocites to the motors
        self.do_motion()

    def run(self):
        # spin until user says to quit
        rospy.spin()

        # if Ctrl-C then exit the spin
        rospy.loginfo("peace homie")


if __name__ == '__main__':
    try:
        ctrl = Controller()
        ctrl.run()
    except rospy.ROSInterruptException:
        pass
