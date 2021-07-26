import os
import cv2
from termcolor import colored


os.system("color")
# LineBreaker
def breaker(num=50, char="*"):
    print(colored("\n" + num*char + "\n", color="magenta"))


# Custom Print Function
def myprint(text, color, on_color=None):
    print(colored(text, color=color, on_color=on_color))


# CLAHE Preprocessing (Cliplimit: 2.0, TileGridSize: (2, 2))
def clahe_equ(image):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
    for i in range(3):
        image[:, :, i] = clahe.apply(image[:, :, i])
    return image


# Webcam Feed Attributes
CAM_WIDTH, CAM_HEIGHT, FPS, ID, WAIT_DELAY = 640, 360, 30, 0, 1
MIN_CROP_WIDTH, MIN_CROP_HEIGHT = 10, 10
