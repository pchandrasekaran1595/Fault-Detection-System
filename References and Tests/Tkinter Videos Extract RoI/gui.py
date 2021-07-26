import tkinter as tk

import utils as u
from VideoFrame import VideoFrame
from VideoSetup import Video

# ******************************************************************************************************************** #

# Application Wrapper
class Wrapper(object):
    def __init__(self, master):
        V = Video(device_id=u.ID, width=u.CAM_WIDTH, height=u.CAM_HEIGHT, fps=u.FPS)

        VideoWidget = VideoFrame(master, V=V)
        VideoWidget.pack()
        VideoWidget.start()

# ******************************************************************************************************************** #

def app():
    root = tk.Tk()
    W, H = int(1.25 * u.CAM_WIDTH), int(1.25 * u.CAM_HEIGHT)
    root.geometry("{}x{}".format(W, H))
    Wrapper(root)
    root.mainloop()

# ******************************************************************************************************************** #
