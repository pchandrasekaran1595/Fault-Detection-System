import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk

import utils as u

# ******************************************************************************************************************** #

class VideoFrame(tk.Frame):
    def __init__(self, master, V=None, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)

        self.master = master
        self.V = V
        self.start_point = None
        self.end_point = None
        self.isCropping = False

        self.canvas = tk.Canvas(self, width=640, height=480, background="black")
        self.canvas.pack()

        # Bind Mouse Button-1 Press to Start Event Handler
        self.canvas.bind("<Button-1>", self.start_handle)

        # Bind Mouse motion to End Event Handler
        self.canvas.bind("<B1-Motion>", self.end_handle)

        # Bind Mouse Button-1 Release to Capture Event Handler
        self.canvas.bind("<ButtonRelease-1>", self.cap)

        self.count = len(os.listdir(u.IMAGE_PATH)) + 1
        self.frame = None
        self.image = None
        self.id = None

        # Delay telling how often to refresh the frame
        self.delay = 15

     # Function to start the Video Capture
    def start(self):
        self.V.start()
        self.update()
    
    # Function to update the canvas every 15 ms
    def update(self):
        ret, frame = self.V.get_frame()
        self.frame = frame.copy()
        if self.start_point is not None and self.end_point is not None and self.isCropping:
            frame = cv2.rectangle(frame, pt1=self.start_point, pt2=self.end_point, color=(255, 255, 255), thickness=2)
        if ret:
            self.image = ImageTk.PhotoImage(Image.fromarray(frame))
            self.canvas.create_image(0, 0, anchor="nw", image=self.image)
        self.id = self.after(self.delay, self.update)

    # Function to stop the Video Capture
    def stop(self):
        if self.id:
            self.after_cancel(self.id)
            self.id = None
            self.V.stop()
    
    # Event handler to record the start point of the crop
    def start_handle(self, event):
        self.end_point = None
        self.isCropping = True
        self.start_point = [event.x, event.y]
        
    # Event handler to record the end point of the crop
    def end_handle(self, event):
        self.end_point = [event.x, event.y]
    
    # Event handler to save the cropped image
    def cap(self, event):
        self.frame = self.frame[self.start_point[1]:self.end_point[1], self.start_point[0]:self.end_point[0], :]
        cv2.imwrite(os.path.join(u.IMAGE_PATH, "Snapshot_{}.png".format(self.count)), cv2.cvtColor(src=self.frame, code=cv2.COLOR_RGB2BGR))
        self.count += 1
        self.isCropping = False

# ******************************************************************************************************************** #