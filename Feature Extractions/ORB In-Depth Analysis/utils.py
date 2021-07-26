"""
    Scipt that hold constansts, fucntions used across various scripts, etc.
"""

import os
from termcolor import colored

# Linebreaker to improve readability of output. Can be used wherever necessary.
os.system("color")
def breaker(num=50, char="*"):
    print(colored("\n" + num*char + "\n", color="red"))


def myprint(text, color, on_color=None):
    print(colored(text, color=color, on_color=on_color))


# Setting up self-aware Image Directory
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "Images")

# Webcam Feed Attributes
CAM_WIDTH, CAM_HEIGHT, FPS, ID = 640, 360, 30, 0
