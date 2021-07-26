import os
import sys
import cv2
import platform
import tkinter as tk
from PIL import Image, ImageTk

# Webcam Canvas Attributes
CAM_WIDTH, CAM_HEIGHT = 640, 360

# ******************************************************************************************************************** #

# Setting up self-aware Image Capture Directory
SAVE_PATH = os.path.join(os.getcwd(), "Captures")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# ******************************************************************************************************************** #

class Video(object):
    def __init__(self, device_id=None, width=None, height=None, fps=30):
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
        if platform.system() != "Windows":
            self.cap = cv2.VideoCapture(self.device_id)
        else:
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
    
    # Releases the capture object
    def stop(self):
        if self.cap.isOpened():
            self.cap.release()
        
    # Gets a frame from the capture object
    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return ret, cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
            else:
                return ret, None

# ******************************************************************************************************************** #

# Tkinter Frame that handles the Video Feed
class VideoFrame(tk.Frame):
    def __init__(self, master, V=None, w=None, h=None, *args, **kwargs):
        """
            master: master widget upon which this works
            V: Video Capture Object
            w: Width of the Canvas Used
            h: height of the Canvas Used
        """
        tk.Frame.__init__(self, master, *args, **kwargs)

        self.master = master

        self.V = V
        self.image = None

        self.canvas = tk.Canvas(self, width=w, height=h, background="black")
        self.canvas.pack()

        self.delay = 15
        self.id = None

    # Function to update the canvas every 15 ms
    def update(self):
        ret, frame = self.V.get_frame()
        if ret:
            self.image = ImageTk.PhotoImage(Image.fromarray(frame))
            self.canvas.create_image(0, 0, anchor="nw", image=self.image)
        self.id = self.after(self.delay, self.update)
    
    # Function to start the Video Capture
    def start(self):
        self.update()

    # Function to stop the Video Capture
    def stop(self):
        if self.id:
            self.after_cancel(self.id)
            self.id = None

# ******************************************************************************************************************** #

# Tkinter Frame that handles the Buttons
class ActionFrame(tk.Frame):
    def __init__(self, master, V=None, path=None, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        """
            master: master widget upon which this works
            V: Video Capture Object
            path: Path to which to save the image capture
        """

        self.master = master
        self.V = V
        self.path = path
        self.count = 1

        button_height, button_width = 2, 20

        # Capture Button Setup
        self.CaptureButton = tk.Button(self, text="Capture", background="#00E8FF", activebackground="#86F4FF", foreground="black",
                                       width=button_width, height=button_height, relief="raised", command=self.capture)
        self.CaptureButton.grid(row=0, column=0)

        # Quit Button Setup
        self.QuitButton = tk.Button(self, text="Quit", background="#FF0000", activebackground="#FCAEAE", foreground="black",
                                    width=button_width, height=button_height, relief="raised", command=self.do_quit)
        self.QuitButton.grid(row=0, column=1)
    
    # Capture Button Callback
    def capture(self):
        ret, frame = self.V.get_frame()
        if ret:
             cv2.imwrite(os.path.join(self.path, "Snapshot_{}.png".format(self.count)), cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB))
             self.count += 1

    # Quit Button Callback
    def do_quit(self):
        self.master.destroy()

# ******************************************************************************************************************** #

# Wrapper for all the Tkinter Frames
class Main(object):
    def __init__(self, master, id=None, w=None, h=None, path=None, num_of_frames=None):

        V = Video(id, w, h)
        V.start()
        VideoWidget = VideoFrame(master, V=V, w=w, h=h)
        VideoWidget.pack()
        VideoWidget.start()
        ActionWidget = ActionFrame(master, V=V, path=path)
        ActionWidget.pack()

# ******************************************************************************************************************** #

"""
    CLI Arguments:
        1. --id : Device ID used for Video Capture (default: 0)
        2. --w  : Width fo the Capture  Frame (Default: 640)
        3. --h  : Height of the Capture Frame (Default: 360)
"""

def app():
    args_1 = "--id"
    args_2 = "--w"
    args_3 = "--h"

    # Default CLI argument values
    device_id = 0
    w = 640
    h = 360

    if args_1 in sys.argv:
        device_id = int(sys.argv[sys.argv.index(args_1) + 1])
    if args_2 in sys.argv:
        w = int(sys.argv[sys.argv.index(args_2) + 1])
    if args_3 in sys.argv:
        h = int(sys.argv[sys.argv.index(args_3) + 1])
    
    # Open a new Tkinter Window
    root = tk.Tk()

    # Setting up the root window size
    ww, wh = int(1.05*w), int(1.175*h)   
    root.geometry("{}x{}".format(ww, wh))

    # Setting up the root window title
    root.title("Capture Application")

    # Calling the Application Wrapper
    Main(root, device_id, w, h, SAVE_PATH)

    # Start
    root.mainloop()

# ******************************************************************************************************************** #