import cv2
  
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
from threading import Thread
img = None
KeepAlive = True


def show_imgage():
    while KeepAlive:
        if img is not None:
            cv2.imshow("drone", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


recorder = None
n = 0

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    img = frame.copy()
    # Display the resulting frame
    #cv2.imshow('frame', frame)
    
    if n == 0:
        recorder = Thread(target=show_imgage)
        recorder.start()
    # recorder = Thread(target=show_imgage)
    # recorder.start()

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
recorder.join()
cv2.destroyAllWindows()