import tkinter as tk

import utils as u
from VideoSetup import Video
from VideoFrame import VideoFrame
from ButtonFrame import ButtonFrame

# ******************************************************************************************************************** #

# Wrapper for all the Tkinter Frames
class Application(object):
    def __init__(self, master, V=None):
        
        self.master = master
        self.V = V

        VideoWidget = VideoFrame(master, V=self.V)
        VideoWidget.pack(side="top")
        VideoWidget.start()

        ButtonWidget = ButtonFrame(master, VideoWidget=VideoWidget)
        ButtonWidget.pack(side="bottom")

# ******************************************************************************************************************** #

def app():
    # Open a new Tkinter Window
    root = tk.Tk()

     # Setting up the root window attributes
    rw, rh = 256, 256
    root.geometry("{}x{}".format(rw, rh))
    root.title("Root Window")

    # Setting up the Top Level window attributes
    window = tk.Toplevel()
    window.geometry("{}x{}".format(u.MAIN_WIDTH, u.MAIN_HEIGHT))
    window.title("Application Window")
    
    # Setting up Video Capture Object
    V = Video(id=u.ID, width=u.CAM_WIDTH, height=u.CAM_HEIGHT, fps=u.FPS)

    # Calling the Application Wrapper
    Application(window, V=V)

    # Start
    root.mainloop()
    
# ******************************************************************************************************************** #
