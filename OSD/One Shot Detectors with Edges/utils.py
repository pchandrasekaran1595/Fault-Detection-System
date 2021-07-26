"""
    Scipt that hold constansts, fucntions used across various scripts, etc.
"""

import os
import cv2
from termcolor import colored
from imgaug import augmenters

# Linebreaker to improve readability of output. Can be used wherever necessary.
os.system("color")
def breaker(num=50, char='*'):
    print(colored("\n" + num*char + "\n", color="red"))


# Setting up self-aware Image Directory
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "Images")
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)


# Center Crop Preprocessing (Reshape to 256x256, then center crop to 224x224)
def preprocess(image, change_color_space=True):
    if change_color_space:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    image = cv2.resize(src=image, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    return image[cy - 112:cy + 112, cx - 112:cx + 112, :]


# Webcam Feed Attributes
CAM_WIDTH, CAM_HEIGHT, FPS, ID = 640, 360, 30, 0

AUGMENT = augmenters.pillike.FilterFindEdges(seed=0)