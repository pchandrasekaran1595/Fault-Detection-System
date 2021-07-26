"""
    Scipt that hold constansts, fucntions used across various scripts, etc.
"""

import os
import cv2
from termcolor import colored
from imgaug import augmenters

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


# Center Crop (Resize to 256x256, then center crop the 224x224 region)
def preprocess(image, change_color_space=True):
    if change_color_space:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    image = cv2.resize(src=image, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    return image[cy - 112:cy + 112, cx - 112:cx + 112, :]


# Setting up self-aware Image Directory
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "Images")
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)


# Webcam Feed Attributes
CAM_WIDTH, CAM_HEIGHT, FPS, ID = 640, 360, 30, 0

AUGMENT = augmenters.pillike.FilterFindEdges(seed=0)