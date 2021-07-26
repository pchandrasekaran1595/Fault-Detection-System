import cv2
import platform


class Video(object):
    def __init__(self, device_id=None, width=None, height=None, fps=None):
        """
            device_id: (int) Device ID of the capture device (can be set via command line)
            width: (int) Width of the Capture Frame
            height: (int) Height of the Capture Frame
            fps: (int) FPS of the Capture Object
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
    
    # Setting up the capture object
    def start(self):
        if platform.system() != 'Windows':
            self.cap = cv2.VideoCapture(self.device_id)
        else:
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    # Gets a frame from the capture object   
    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return ret, cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
            else:
                return ret, None

    # Releases the capture object
    def stop(self):
        if self.cap.isOpened():
            self.cap.release()
