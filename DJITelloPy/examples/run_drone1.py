import cv2
#import pygame
from djitellopy import Tello
import imutils
import queue
import threading
import sys

# Initialize the Tello drone
tello = Tello()
tello.connect()
#tello.takeoff()
#tello.move_up(50)
# Start the video stream
tello.streamon()

# Connect to the video stream using OpenCV
cap = cv2.VideoCapture('udp://0.0.0.0:11111')

run_process = True
command_data = queue.Queue()

def capture_video():
    global run_process
    while True and run_process:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Drone Video", frame)
        key = cv2.waitKey(1) & 0xff
        if key == 27: # ESC
            run_process = False
            break
        elif key == ord('w'):
            command_data.put("go_up")
        elif key == ord('r'):
            command_data.put("rotate")
        elif key == ord('l'):
            command_data.put("land")

    tello.streamoff()
    tello.land()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)


def run_drone(command_data):
    global run_process,tello
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

# Continuously receive and display the video frames
# while True:
#     ret, frame = cap.read()
#     frame = imutils.resize(frame, width=512, height=512)
#     if ret is None:
#         continue
#     cv2.imshow('Tello Video Stream', frame)
#     key = cv2.waitKey(1) & 0xff
#     if key == 27: # ESC
#         break
#     elif key == ord('w'):
#         print("move foward")
#     elif key == ord('s'):
#         tello.move_back(30)
#     elif key == ord('a'):
#         tello.move_left(30)
#     elif key == ord('d'):
#         tello.move_right(30)
#     elif key == ord('e'):
#         tello.rotate_clockwise(30)
#     elif key == ord('q'):
#         tello.rotate_counter_clockwise(30)
#     elif key == ord('r'):
#         tello.move_up(30)
#     elif key == ord('f'):
#         tello.move_down(30)

# Start the first thread
first_thread = threading.Thread(target=capture_video)
first_thread.start()

# Start the second thread
run_d = threading.Thread(target=run_drone, args=(command_data,))
run_d.start()


# Land the drone and release the video stream
#tello.land()
# tello.streamoff()
# cap.release()
# cv2.destroyAllWindows()