import queue
import threading
import cv2,sys
# Create a queue to store the data
data_queue = queue.Queue()
run_process = True

cap = cv2.VideoCapture(0)

# Function that will run in the first thread
def first_thread_func(data_queue):
    while True and run_process:
        ret, frame = cap.read()
        if ret:
            data_queue.put(frame)
    sys.exit(0)

# Function that will run in the second thread
def second_thread_func(data_queue):
    global run_process
    # Get the data from the queue
    while True and run_process:
        data = data_queue.get()
        cv2.imshow("name", data)
        if cv2.waitKey(1) == ord("q"):
            run_process = False
            break
   
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)
    

# Start the first thread
first_thread = threading.Thread(target=first_thread_func, args=(data_queue,))
first_thread.start()

# Start the second thread
second_thread = threading.Thread(target=second_thread_func, args=(data_queue,))
second_thread.start()

# second_thread.join()
# first_thread.join()

