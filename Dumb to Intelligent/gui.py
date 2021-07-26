import os
import sys
import cv2
import ctypes
import torch
import platform
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from time import time 

import Models
import utils as u
from MakeData import make_data 
from Train import trainer

_, batch_size, lr, wd = Models.build_siamese_model()
screen_resolution = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))

# ******************************************************************************************************************** #

# Inference Helper
def __help__(frame=None, model=None, show_prob=True, fea_extractor=None):
    disp_frame = frame.copy()
    h, w, _ = frame.shape

    frame = u.preprocess(frame, False)
    
    with torch.no_grad():
        features = u.normalize(fea_extractor(u.FEA_TRANSFORM(frame).to(u.DEVICE).unsqueeze(dim=0)))
        y_pred = torch.sigmoid(model(features))[0][0].item()

    if show_prob:
        if y_pred >= u.upper_bound_confidence:
            cv2.putText(img=disp_frame, text="Match, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.GUI_GREEN, thickness=2)
            # cv2.rectangle(img=disp_frame, 
            #               pt1=(int(w/2) - 100, int(h/2) - 100), 
            #               pt2=(int(w/2) + 100, int(h/2) + 100), 
            #               color=u.GUI_GREEN, thickness=2)
        elif u.lower_bound_confidence <= y_pred <= u.upper_bound_confidence:
            cv2.putText(img=disp_frame, text="Possible Match, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.GUI_ORANGE, thickness=2)
            # cv2.rectangle(img=disp_frame, 
            #               pt1=(int(w/2) - 100, int(h/2) - 100), 
            #               pt2=(int(w/2) + 100, int(h/2) + 100), 
            #               color=u.GUI_ORANGE, thickness=2)
        else:
            cv2.putText(img=disp_frame, text="No Match, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.GUI_RED, thickness=2)
            # cv2.rectangle(img=disp_frame, 
            #               pt1=(int(w/2) - 100, int(h/2) - 100), 
            #               pt2=(int(w/2) + 100, int(h/2) + 100), 
            #               color=u.GUI_RED, thickness=2)
    else:
        if y_pred >= u.lower_bound_confidence:
            cv2.putText(img=disp_frame, text="Match", org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=(0, 255, 0), thickness=2)
            # cv2.rectangle(img=disp_frame, 
            #               pt1=(int(w/2) - 100, int(h/2) - 100), 
            #               pt2=(int(w/2) + 100, int(h/2) + 100), 
            #               color=u.GUI_GREEN, thickness=2) 
        elif u.lower_bound_confidence <= y_pred <= u.upper_bound_confidence:
            cv2.putText(img=disp_frame, text="Possible Match", org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.GUI_ORANGE, thickness=2)
            # cv2.rectangle(img=disp_frame, 
            #               pt1=(int(w/2) - 100, int(h/2) - 100), 
            #               pt2=(int(w/2) + 100, int(h/2) + 100), 
            #               color=u.GUI_ORANGE, thickness=2)
        else:
            cv2.putText(img=disp_frame, text="No Match", org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.GUI_RED, thickness=2)
            # cv2.rectangle(img=disp_frame, 
            #               pt1=(int(w/2) - 100, int(h/2) - 100), 
            #               pt2=(int(w/2) + 100, int(h/2) + 100), 
            #               color=u.GUI_RED, thickness=2)
    return disp_frame

# ******************************************************************************************************************** #

class Video(object):
    def __init__(self, id=None, width=None, height=None, fps=None):
        self.id = id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
    
    def start(self):
        if platform.system() != 'Windows':
            self.cap = cv2.VideoCapture(self.id)
        else:
            self.cap = cv2.VideoCapture(self.id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
    
    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return ret, cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
            else:
                return ret, None

    def stop(self):
        if self.cap.isOpened():
            self.cap.release()


class VideoFrame(tk.Frame):
    def __init__(self, master, V=None, model=None, part_name=None, isResult=False, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)

        self.master = master
        self.V = V
        self.image = None
        self.isResult = isResult
        self.part_name = part_name
        self.model = model

        # If VideoFrame is in Result Mode; Default is None
        if self.isResult:
            self.model_path = os.path.join(os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Checkpoints"), "State.pt")
            self.model.load_state_dict(torch.load(self.model_path, map_location=u.DEVICE)["model_state_dict"])
            self.model.eval()
            self.model.to(u.DEVICE)

        self.canvas = tk.Canvas(self, width=u.CAM_WIDTH, height=u.CAM_HEIGHT, background="black")
        self.canvas.pack()

        self.delay = 15
        self.id = None
    
    def start(self):
        self.V.start()
        self.update()
    
    def update(self):
        ret, frame = self.V.get_frame()

        if not self.isResult:
            frame = u.clahe_equ(frame)
            if ret:
                # h, w, _ = frame.shape
                # frame = cv2.rectangle(img=frame, pt1=(int(w/2) - 100, int(h/2) - 100), pt2=(int(w/2) + 100, int(h/2) + 100), color=(255, 255, 255), thickness=2)
                self.image = ImageTk.PhotoImage(Image.fromarray(frame))
                self.canvas.create_image(0, 0, anchor="nw", image=self.image)
                self.id = self.after(self.delay, self.update)
            else:
                return
        else:
            if ret:
                frame = u.clahe_equ(frame)
                frame = __help__(frame=frame, model=self.model, 
                                 show_prob=False, fea_extractor=Models.fea_extractor)
                self.image = ImageTk.PhotoImage(Image.fromarray(frame))
                self.canvas.create_image(0, 0, anchor="nw", image=self.image)
                self.id = self.after(self.delay, self.update)
            else:
                return
        
    def stop(self):
        if self.id:
            self.after_cancel(self.id)
            self.id = None
            self.V.stop()
    
# ******************************************************************************************************************** #

class ImageFrame(tk.Frame):
    def __init__(self, master, imgfilepath, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)

        self.master = master
        self.image = None

        self.canvas = tk.Canvas(self, width=u.CAM_WIDTH, height=u.CAM_HEIGHT, background="black")
        self.canvas.pack()

        if imgfilepath:
            self.image = cv2.cvtColor(src=cv2.imread(imgfilepath, cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB)
            self.image = cv2.resize(src=self.image, dsize=(u.CAM_WIDTH, u.CAM_HEIGHT), interpolation=cv2.INTER_AREA)
            self.image = ImageTk.PhotoImage(Image.fromarray(self.image))
            self.canvas.create_image(0, 0, anchor="nw", image=self.image)

# ******************************************************************************************************************** #

class ButtonFrame(tk.Frame):
    def __init__(self, master, VideoWidget=None, ImageWidget=None, model=None, part_name=None, isFirstTimeRun=None, *args, **kwargs):
        tk.Frame.__init__(self, master, width=150, background="#2C40D1", *args, **kwargs)

        self.master = master
        self.VideoWidget = VideoWidget
        self.ImageWidget = ImageWidget
        self.widget_height = 3
        self.widget_width = 25
        self.mdoel = model
        self.isFirstTimeRun = isFirstTimeRun

        self.part_name = part_name
        self.countn, self.countp = 1, 1

        self.model = model

        # Label
        self.label = tk.Label(self, text="Component/Part Name", 
                              background="gray", foreground="black", 
                              width=self.widget_width, height=self.widget_height,
                              relief="raised")
        self.label.grid(row=0, column=0)

        # Entry
        self.entry = tk.Entry(self, background="white", foreground="black",
                              selectbackground="blue", selectforeground="white", 
                              width=self.widget_width, relief="sunken")
        self.entry.grid(row=0, column=1)

        # Add to Positive
        self.posButton = tk.Button(self, text="Add to Positive",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#DFDFDC", activebackground="#F6F6F4", foreground="black",
                                     relief="raised", command=self.do_pos)
        self.posButton.grid(row=1, column=0)

        # Add to Negative
        self.negButton = tk.Button(self, text="Add to Negative",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#505050", activebackground="#717171", foreground="black",
                                     relief="raised", command=self.do_neg)
        self.negButton.grid(row=1, column=1)

        # Train
        self.trainButton = tk.Button(self, text="Train",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#45FE33", activebackground="#8EFF83", foreground="black",
                                     relief="raised", command=self.do_train)
        self.trainButton.grid(row=2, column=0)

        # Realtime Application
        self.rtAppButton = tk.Button(self, text="Application",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#00FFD4", activebackground="#8AFFEB", foreground="black",
                                     relief="raised", command=self.do_rtapp)
        self.rtAppButton.grid(row=2, column=1)

        # Reset
        self.resetButton = tk.Button(self, text="Reset",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#FFC500", activebackground="#FFDF84", foreground="black",
                                     relief="raised", command=self.do_reset)
        self.resetButton.grid(row=3, column=0)

        # Quit
        self.quitButton = tk.Button(self, text="Quit",
                                    width=self.widget_width, height=self.widget_height, 
                                    background="red", activebackground="#FCAEAE", foreground="black",
                                    relief="raised", command=self.do_quit)
        self.quitButton.grid(row=3, column=1)

    # Callback handling adding images to Positive Diretory
    def do_pos(self):
        self.part_name = self.entry.get()
        if self.part_name:
            self.path = os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Positive")
        else:
            messagebox.showerror(title="Value Error", message="Enter a valid input")
            return
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.countp = len(os.listdir(self.path)) + 1
        ret, frame = self.VideoWidget.V.get_frame()
        frame = u.clahe_equ(frame)

        if ret:
            cv2.imwrite(os.path.join(self.path, "Snapshot_{}.png".format(self.countp)), cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB))
            self.countp += 1

    # Callback handling adding images to Negative Diretory
    def do_neg(self):
        self.part_name = self.entry.get()
        if self.part_name:
            self.path = os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Negative")
        else:
            messagebox.showerror(title="Value Error", message="Enter a valid input")
            return
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.countn = len(os.listdir(self.path)) + 1
        ret, frame = self.VideoWidget.V.get_frame()
        frame = u.clahe_equ(frame)

        if ret:
            cv2.imwrite(os.path.join(self.path, "Snapshot_{}.png".format(self.countn)), cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB))
            self.countn += 1
    
    # Callback handling training of objects
    def do_train(self):
        self.part_name = self.entry.get()
        if self.part_name:
            u.breaker()

            # path = os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Checkpoints")
            # names = os.listdir(path)
            # if "State.pt" in names:
            #     self.model_path = os.path.join(os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Checkpoints"), "State.pt")
            #     self.model.load_state_dict(torch.load(self.model_path, map_location=u.DEVICE)["model_state_dict"])

            self.VideoWidget.stop()
            self.master.iconify()

            u.myprint("Generating Feature Vector Data ...", "green")
            start_time = time()
            make_data(part_name=self.part_name, cls="Positive", num_samples=u.num_samples, fea_extractor=Models.fea_extractor, roi_extractor=Models.roi_extractor)
            make_data(part_name=self.part_name, cls="Negative", num_samples=u.num_samples, fea_extractor=Models.fea_extractor, roi_extractor=Models.roi_extractor)
            u.myprint("\nTime Taken [{}] : {:.2f} minutes".format(2*u.num_samples, (time()-start_time)/60), "green")
            trainer(part_name=self.part_name, model=self.model, epochs=u.epochs, lr=lr, wd=wd, batch_size=batch_size, early_stopping=u.early_stopping_step, fea_extractor=Models.fea_extractor)

            self.VideoWidget.start()
            self.master.state("zoomed")

            u.breaker()
        else:
            messagebox.showerror(title="Value Error", message="Enter a valid input")
            return
    
    # Callback handling realtime application start
    def do_rtapp(self):
        self.part_name = self.entry.get()
        if self.part_name:
            self.master.destroy()
            setup(part_name=self.part_name, model=self.model, imgfilepath=os.path.join(os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Positive"), "Snapshot_1.png"), isResult=True)
        else:
            messagebox.showerror(title="Value Error", message="Enter a valid input")
            return
    
    # Callback resetting the application
    def do_reset(self):
        self.VideoWidget.V.stop()
        self.master.destroy()
        model, _, _, _ = Models.build_siamese_model(embed=u.embed_layer_size)
        setup(model=model)
    
    # Callback quitting the application
    def do_quit(self):
        self.VideoWidget.V.stop()
        self.master.master.destroy()

# ******************************************************************************************************************** #

# Wrapper
class Application():
    def __init__(self, master, V=None, part_name=None, model=None, imgfilepath=None, isResult=False):

        VideoWidget = VideoFrame(master, V=V, model=model, part_name=part_name, isResult=isResult)
        VideoWidget.pack(side="left")
        VideoWidget.start()
        ImageWidget = ImageFrame(master, imgfilepath=imgfilepath)
        ImageWidget.pack(side="right")
        ButtonWidget = ButtonFrame(master, VideoWidget=VideoWidget, ImageWidget=ImageWidget, model=model, part_name=part_name)
        ButtonWidget.pack(side="bottom")

# ******************************************************************************************************************** #

# Top level window setup and Application start
def setup(part_name=None, model=None, imgfilepath=None, isResult=False):
    window = tk.Toplevel()
    window.title("Application")
    window.geometry("{}x{}".format(screen_resolution[0], screen_resolution[1]))
    window.state("zoomed")
    w_canvas = tk.Canvas(window, width=screen_resolution[0], height=screen_resolution[1], bg="#40048C")
    w_canvas.place(x=0, y=0)
    Application(window, V=Video(id=u.device_id, width=u.CAM_WIDTH, height=u.CAM_HEIGHT, fps=u.FPS), 
                part_name=part_name, model=model, imgfilepath=imgfilepath, isResult=isResult)


# ******************************************************************************************************************** #

# Building the GUI Application; contains basic CLI arguments
def app():
    args_1 = "--num-samples"
    args_2 = "--embed"
    args_3 = "--epochs"
    args_4 = "--lower"
    args_5 = "--upper"
    args_6 = "--early"

    # CLI Argument Handling
    if args_1 in sys.argv:
        u.num_samples = int(sys.argv[sys.argv.index(args_1) + 1])
    if args_2 in sys.argv:
        u.embed_layer_size = int(sys.argv[sys.argv.index(args_2) + 1])
    if args_3 in sys.argv:
        u.epochs = int(sys.argv[sys.argv.index(args_3) + 1])
    if args_4 in sys.argv:
        u.lower_bound_confidence = float(sys.argv[sys.argv.index(args_4) + 1])        
    if args_5 in sys.argv:
        u.upper_bound_confidence = float(sys.argv[sys.argv.index(args_5) + 1]) 
    if args_6 in sys.argv:
        u.early_stopping_step = int(sys.argv[sys.argv.index(args_6) + 1]) 

    root = tk.Tk()
    rw, rh = 256, 256
    root.geometry("{}x{}".format(rw, rh))
    root.title("Root Window")
    root.iconify()

    model, _, _, _ = Models.build_siamese_model(embed=u.embed_layer_size)
    setup(model=model)
    
    # Start
    root.mainloop()

# ******************************************************************************************************************** #