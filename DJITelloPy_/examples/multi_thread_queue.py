import threading
import time 
#from Queue import Queue
from multiprocessing import Queue

print_lock = threading.Lock()

q = Queue()

class MyThread(threading.Thread):
    def __init__(self, queue, args=(), kwargs=None):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.queue = queue
        self.daemon = True
        self.receive_messages = args[0]

    def run(self):
        print (threading.current_thread(), self.receive_messages)
        val = self.queue.get()
        self.do_thing_with_message(val)

    def do_thing_with_message(self, message):
        if self.receive_messages:
            with print_lock:
                print (threading.current_thread(), "Received {}".format(message))


class MyThread2(threading.Thread):
    def __init__(self, queue, args=(), kwargs=None):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.queue = queue
        self.daemon = True
        self.receive_messages = args[0]

    def run(self):
        print (threading.current_thread(), self.receive_messages)
        val = self.queue.get()
        self.do_thing_with_message(val)

    def do_thing_with_message(self, message):
        if self.receive_messages:
            with print_lock:
                print (threading.current_thread(), "Received {}".format(message))


class MyThread3(threading.Thread):
    def __init__(self, queue, args=(), kwargs=None):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.queue = queue
        self.daemon = True
        self.receive_messages = args[0]

    def run(self):
        print (threading.current_thread().getName(), self.receive_messages)
        val = self.queue.get()
        self.do_thing_with_message(val)

    def do_thing_with_message(self, message):
        if self.receive_messages:
            with print_lock:
                print (threading.current_thread().getName(), "Received {}".format(message))

if __name__ == '__main__':
    threads = []
    for t in range(10):
        threads.append(MyThread(q, args=(t % 2 == 0,)))
        threads[t].start()
        time.sleep(0.1)

    for t in threads:
        t.queue.put("Print this!")