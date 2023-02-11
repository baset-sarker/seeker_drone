import queue
import threading
import cv2,sys
# Create a queue to store the data
data_queue = queue.Queue()
run_process = True

#cap = cv2.VideoCapture(0)


from djitellopy import Tello
import cv2, math, time

from threading import Thread

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()


# Function that will run in the first thread
def capture_video():
    global run_process
    while True and run_process:
        frame = frame_read.frame
        while True and run_process:
            frame = frame_read.frame
            cv2.imshow("name", frame)
            if cv2.waitKey(1) == ord("q"):
                run_process = False
            if cv2.waitKey(1) == ord("s"):
                data_queue.put("rotate")
                break

    sys.exit(0)


# Function that will run in the second thread
def second_thread_func(data_queue):
    global run_process
    while True and run_process:
        data = data_queue.get()
        if data == "rotate":
            print("rotate")
            #tello.rotate_counter_clockwise(10)
   
    # global run_process
    # # Get the data from the queue
    # while True and run_process:
    #     data = data_queue.get()
    #     cv2.imshow("name", data)
    #     if cv2.waitKey(1) == ord("q"):
    #         run_process = False
    #         break
   
    # cap.release()
    # cv2.destroyAllWindows()
    sys.exit(0)
    

# Start the first thread
first_thread = threading.Thread(target=capture_video)
first_thread.start()

# Start the second thread
second_thread = threading.Thread(target=second_thread_func, args=(data_queue,))
second_thread.start()

#tello.takeoff()

# second_thread.join()
# first_thread.join()

