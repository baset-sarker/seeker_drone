import cv2
import threading

class FrameThread(threading.Thread):
    def __init__(self, name, cap):
        super().__init__()
        self.name = name
        self.cap = cap

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            cv2.imshow(self.name, frame)
            if cv2.waitKey(1) == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    thread = FrameThread("Frame", cap)
    thread.start()
    

    i = 0
    while(i < 100):
        print("main thread")
        i = i + 1
        
    
    thread.join()
