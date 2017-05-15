import con
import serial


def main():
    port = "/dev/ttyACM0"
    baud = 9600
    joy = con.hci_init()

    ser = serial.Serial(port, baud, timeout=1)

    if ser.isOpen():
        print ser.name + 'is open...'
    base = 2047
    # ser.write("1,start\r\n")
    # ser.write("2,start\r\n")
    r_speed = 0
    l_speed = 0
    l_prev = 0
    r_prev = 0
    while True:

        if con.hci_button(joy,5):
                tol = 100
                r_speed,l_speed = con.hci_input(joy)
                r_speed = int(r_speed*base)
                l_speed = int(l_speed*base)
                    # if self.state == "following":
                    #Cap on change in velocity, smoother ride
                if r_speed - r_prev > tol:
                    r_speed = r_prev + tol
                if r_speed - r_prev < -(2*tol):
                    r_speed = r_prev - (2*tol)
                if l_speed - l_prev > tol:
                    l_speed = l_prev + tol
                if l_speed - l_prev < -(2*tol):
                    l_speed = l_prev - (2*tol)


                ser.write("M1:"+str(r_speed)+'\r\n')
                print("M1:"+str(r_speed))
                ser.write("M2:"+str(l_speed)+'\r\n')
                print("M2:"+str(l_speed))
                l_prev = l_speed
                r_prev = r_speed

        ser.write("M1:"+str(0)+'\r\n')
        ser.write("M2:"+str(0)+'\r\n')

if __name__ == '__main__':
    main()
