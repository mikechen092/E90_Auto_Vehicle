# This code was taken from Noah Weinthal and Neil MacFarland E90
# with help from David Ranshous

import pygame
from pygame.locals import *
import sys

def hci_init():
    pygame.init()
    j = pygame.joystick.Joystick(0)
    j.init()
    print('Initialized Joystick : {}'.format(j.get_name()))
    return j

def hci_input(j):
    pygame.event.pump()

    # Used to read input from the two joysticks
    # print "number of axes: ", j.get_numaxes()
    # print "axis 4: ", j.get_axis(4)
    # alt_throttle = j.get_hat(0) # axis for xbox controller
    steering = j.get_axis(1)
    throttle = j.get_axis(4)        # axis for PS3 controller
    return (steering, -throttle)

def hci_button(j,num):
    pygame.event.pump()
    button = j.get_button(num)
    return button

def main():
    j = hci_init()
    while 1:
        pygame.event.pump()
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                print("key down")
            if len(events) > 0:
                print(events)


if __name__ == '__main__':
    main()
