import time,math
from math import atan2,degrees

def run_command(command,x,tello):
    print(command,x)    
    if command == "takeoff":
        tello.takeoff()
        return True

    if command == "land":
        tello.land()
        return True

    if  command == "move_up":
        """Fly x cm up.
        Arguments:
            x: 20-500
        """
        tello.move_up(x)
        return True
        


    if  command == "move_down":
        """Fly x cm down.
        Arguments:
            x: 20-500
        """
        tello.move_down(x)
        return True
        

    if  command == "move_left":
        """Fly x cm left.
        Arguments:
            x: 20-500
        """
        tello.move_left(x)
        return True
        

    if  command == "move_right":
        """Fly x cm right.
        Arguments:
            x: 20-500
        """
        tello.move_right(x)
        return True
        

    if  command == "move_forward":
        """Fly x cm forward.
        Arguments:
            x: 20-500
        """
        tello.move_forward(x)
        return True
        

    if  command == "move_back":
        """Fly x cm backwards.
        Arguments:
            x: 20-500
        """
        tello.move_back(x)
        return True
        

    if  command == "rotate_clockwise":
        """Rotate x degree clockwise.
        Arguments:
            x: 1-360
        """
        tello.rotate_clockwise(x)
        return True
        

    if  command == "rotate_counter_clockwise":
        """Rotate x degree counter-clockwise.
        Arguments:
            x: 1-3600
        """
        tello.rotate_counter_clockwise(x)
        return True

    if command == "set_speed":
        """Set speed to x cm/s.
        Arguments:
            x: 10-100
        """
        tello.set_speed(x)
        return True

    if  command == "end":
        tello.end()
        return True

    return True

        

def get_area(xywh):
    x, y, w, h = xywh
    return (w - x) * (h - y)

def get_center(xywh):
    x, y, w, h = xywh
    return (int(x),int(y))

def get_distance(p, q):
    return math.dist(p, q)


def get_angle(pointA, pointB):
  changeInX = pointB[0] - pointA[0]
  changeInY = pointB[1] - pointA[1]
  return degrees(atan2(changeInY,changeInX))


def manage_drone(distance, angle, area):
    # check if object is closer than 100 cm
    if distance < 100 and (angle >= 80 and angle <= 100):
       return "land"
         
    
    return 1