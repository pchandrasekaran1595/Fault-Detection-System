import tkinter as tk

# Tkinter Frame that handles the Buttons
class ButtonFrame(tk.Frame):
    def __init__(self, master, VideoWidget=None, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)

        """
            master: master widget upon which this works
            VideoWidget: Video Capture Frame
        """
        self.master = master
        self.VideoWidget = VideoWidget
        self.button_height = 2
        self.button_width = 20

        # Start Button Setup
        self.startButton = tk.Button(self, text="Start",
                                    width=self.button_width, height=self.button_height, 
                                    background="#23EF13", activebackground="#9AF592", foreground="black",
                                    relief="raised", command=self.do_start)
        self.startButton.grid(row=0, column=0)

        # Stop Button Setup
        self.stopButton = tk.Button(self, text="Stop",
                                    width=self.button_width, height=self.button_height, 
                                    background="#FFC500", activebackground="#FFE99E", foreground="black",
                                    relief="raised", command=self.do_stop)
        self.stopButton.grid(row=0, column=1)

        # Quit Button Setup
        self.quitButton = tk.Button(self, text="Quit",
                                    width=self.button_width, height=self.button_height, 
                                    background="red", activebackground="#FCAEAE", foreground="black",
                                    relief="raised", command=self.do_quit)
        self.quitButton.grid(row=0, column=2)
    
    # Start Button Callback
    def do_start(self):
        self.VideoWidget.start()

    # Stop Button Callback
    def do_stop(self):
        self.VideoWidget.stop()
    
    # Quit Button Callback
    def do_quit(self):
        self.master.master.destroy()
