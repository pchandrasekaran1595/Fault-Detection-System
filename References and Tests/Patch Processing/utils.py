import os
import cv2
from termcolor import colored


os.system("color")
def myprint(text, color, on_color=None):
    print(colored(text, color=color, on_color=on_color))


# Linebreaker to improve readability of output. Can be used wherever necessary.
def breaker(num=50, char="*"):
    print(colored("\n" + num*char + "\n", color="blue"))


# Webcam Feed Attributes
CAM_WIDTH, CAM_HEIGHT, FPS, ID, WAIT_DELAY = 640, 360, 30, 0, 1
MIN_CROP_WIDTH, MIN_CROP_HEIGHT = 10, 10
