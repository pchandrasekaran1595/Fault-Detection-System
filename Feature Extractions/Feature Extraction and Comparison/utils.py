"""
    Scipt that hold constansts, fucntions used across various scripts, etc.
"""

import os
import cv2
from termcolor import colored


os.system("color")
# Linebreaker to improve readability of output. Can be used wherever necessary.
def breaker(num=50, char='*'):
    print(colored("\n" + num*char + "\n", color="red"))

# CLAHE Equalization handler
def clahe_equ(image, clipLimit=2.0, TGS=2):
    n_img = image.copy()
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(TGS, TGS))
    for i in range(3):
        n_img[:, :, i] = clahe.apply(n_img[:, :, i])
    return n_img

# Setting up self-aware Image Directory
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "Images")
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

# Webcam Feed Attributes
CAM_WIDTH, CAM_HEIGHT, FPS, ID = 640, 360, 30, 0
