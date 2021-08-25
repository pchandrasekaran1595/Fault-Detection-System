"""
    Scipt that hold constansts, fucntions used across various scripts, etc.
"""

import os
import cv2
import warnings
from termcolor import colored
warnings.filterwarnings("ignore")


# Linebreaker to improve readability of output. Can be used wherever necessary.
os.system("color")
def breaker(num=50, char='*'):
    print(colored("\n" + num*char + "\n", color="red"))


# Custom Print Function
def myprint(text, color, on_color=None):
    print(colored(text, color=color, on_color=on_color))


# CLAHE Equalization handler
def clahe_equ(image, clipLimit=2.0, TGS=2):
    n_img = image.copy()
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(TGS, TGS))
    for i in range(3):
        n_img[:, :, i] = clahe.apply(n_img[:, :, i])
    return n_img


# Center Crop Preprocessing (Reshape to 256x256, then center crop to 224x224)
def preprocess(image, change_color_space=True):
    if change_color_space:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    image = cv2.resize(src=image, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    return image[cy - 112:cy + 112, cx - 112:cx + 112, :]


# Center Crop (Resize to 366x366, then center crop the 320x320 region)
def preprocess_320(image, change_color_space=True):
    if change_color_space:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    image = cv2.resize(src=image, dsize=(366, 366), interpolation=cv2.INTER_AREA)
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    return image[cy - 160:cy + 160, cx - 160:cx + 160, :]


# Setting up self-aware Image Directory
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "Images")
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

# Webcam Feed Attributes
CAM_WIDTH, CAM_HEIGHT, FPS, ID = 640, 360, 30, 0
