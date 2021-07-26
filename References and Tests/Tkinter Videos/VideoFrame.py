import tkinter as tk
from PIL import ImageTk, Image

import utils as u

class VideoFrame(tk.Frame):
    def __init__(self, master, V=None, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)

        """
            master: master widget upon which this works
            V: Video Capture Object
        """

        self.master = master
        self.V = V
        self.image = None

        self.canvas = tk.Canvas(self, width=u.CAM_WIDTH, height=u.CAM_HEIGHT, background="black")
        self.canvas.pack()

        self.delay = 15
        self.id = None
    
    # Function to start the Video Capture
    def start(self):
        self.V.start()
        self.update()
    
    # Function to update the canvas every 15 ms
    def update(self):
        ret, frame = self.V.get_frame()
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
