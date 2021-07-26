import os

# Webcam Feed Attributes
CAM_WIDTH, CAM_HEIGHT, FPS, ID = 640, 480, 30, 0

# Setting up self-aware Image Directory
IMAGE_PATH = os.path.join(os.getcwd(), "Images")
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)