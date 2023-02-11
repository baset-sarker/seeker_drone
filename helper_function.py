import time,math
from math import atan2,degrees

def run_command_to_drone(command_data,tello):
    global run_process
    # Get the data from the queue
    while True and run_process:
        data = command_data.get()
        if data == "go_up":
            tello.takeoff()
        if data == "rotate":
           tello.rotate_clockwise(30) 
        if data == "land":
            tello.land()
        print(data)

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