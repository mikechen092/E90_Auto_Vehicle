import rospy
from std_msgs.msg import String
import getch
import sys, termios, atexit
from select import select
import time
import serial

# save the terminal settings
fd = sys.stdin.fileno()
new_term = termios.tcgetattr(fd)
old_term = termios.tcgetattr(fd)

# new terminal setting unbuffered
new_term[3] = (new_term[3] & ~termios.ICANON & ~termios.ECHO)

# switch to normal terminal
def set_normal_term():
    termios.tcsetattr(fd, termios.TCSAFLUSH, old_term)

# switch to unbuffered terminal
def set_curses_term():
    termios.tcsetattr(fd, termios.TCSAFLUSH, new_term)

def getch():
    return sys.stdin.read(1)

def kbhit():
    dr,dw,de = select([sys.stdin], [], [], 0)
    return dr <> []

def e_stop(ser):
    while 1:
        # if a key is hit
        print("e_brake")
        if kbhit():
            key = getch()
            if key == 'c':
                break
        # else output speed of zero
        ser.write('M1:'+str(0)+'\r\n')

def main():
    # forgot which port it isdD
    atexit.register(set_normal_term)
    set_curses_term()
    # need to check which one it is it changes like wtf
    port = "/dev/ttyACM0"
    baud = 9600

    ser = serial.Serial(port, baud, timeout=1)

    if ser.isOpen():
        print ser.name + 'is open...'

    print("Please press a key to see its value")
    speed = 0
    turnspeed = 0
    # need to send both signals first then can change afterwards
    ser.write('MD:'+str(0)+'\r\n')
    ser.write('MT:'+str(0)+'\r\n')
    while True:

        # print("forward speed: ",speed)
        # print(turnspeed)
        if kbhit(): # <--------
            key = getch()
            if ord(key) == 32:
                speed = 0
                e_stop(ser)
            elif ord(key) == 119: #w
                if speed >= 0 and speed < 2047:
                    speed += 100
                print('executing command: MD:'+str(speed))
                ser.write('MD:'+str(speed)+'\r\n')
            elif ord(key) == 97: #a
                # have it turn at a constant speed ?
                #print('executing command: MT:-512')
                ser.write("M1:2000\r\n")
                ser.write("M2:-50\r\n")
            elif ord(key) == 115: #s
                # print("s")
                if speed > -2047 and speed <= 0:
                    speed -= 100
                print('executing command: MD:'+str(speed))
                ser.write('MD:'+str(speed)+'\r\n')
            elif ord(key) == 100: #d
                # have it turn at a constant rate?
                #print('executing command: MT:512')
                ser.write('M2:2000\r\n')
                ser.write('M1:-50\r\n')
        else:
            print("not hit")
            if speed > 0:
                speed -= 100
                print('executing command: MD:'+str(speed))
                ser.write('MD:'+str(speed)+'\r\n')
            elif speed < 0:
                speed += 100
                print('executing command: MD:'+str(speed))
                ser.write('MD:'+str(speed)+'\r\n')
            ser.write("MT:0\r\n")




        time.sleep(0.04)
        # rate.sleep()

if __name__ == '__main__':
    main()
