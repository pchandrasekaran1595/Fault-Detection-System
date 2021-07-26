import os
import cv2
import numpy as np
import torch
from torchvision import transforms, ops
from termcolor import colored
os.system("color")

DATASET_PATH = os.path.join(os.getcwd(), "Datasets")
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
# DATASET_PATH = os.path.join(os.path.dirname(__file__), "Datasets")

CAM_WIDTH, CAM_HEIGHT, FPS, DELAY = 640, 360, 30, 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ROI_TRANSFORM = transforms.Compose([transforms.ToTensor(), ])
FEA_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIZE = 224
SEED = 0
RELIEF = 25
FEATURE_VECTOR_LENGTH = 2048

GUI_ORANGE = (255, 165, 0)
GUI_RED    = (255, 0, 0)
GUI_GREEN  = (0, 255, 0)

# ****************************************** Default CLI Arguments *************************************************** #
embed_layer_size = 2048
num_samples = 15000
epochs = 1000
lower_bound_confidence = 0.7
upper_bound_confidence = 0.8
device_id = 0
early_stopping_step = 50
# ******************************************************************************************************************** #

def breaker(num=50, char="*"):
    print(colored("\n" + num*char + "\n", color="magenta"))


def myprint(text, color, on_color=None):
    print(colored(text, color=color, on_color=on_color))


def clahe_equ(image):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
    for i in range(3):
        image[:, :, i] = clahe.apply(image[:, :, i])
    return image

def preprocess(image, change_color_space=True):
    if change_color_space:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    image = cv2.resize(src=image, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    return image[cy - 112:cy + 112, cx - 112:cx + 112, :]

# ******************************************************************************************************************** #

def get_box_coordinates(model, transform, image):
    x1, y1, x2, y2 = None, None, None, None

    h, w, _ = image.shape
    temp_image = image.copy()
    temp_image = preprocess(temp_image, change_color_space=False)

    with torch.no_grad():
        output = model(transform(temp_image).to(DEVICE).unsqueeze(dim=0))[0]
    cnts, scrs = output["boxes"], output["scores"]
    if len(cnts) != 0:
        cnts = ops.clip_boxes_to_image(cnts, (SIZE, SIZE))
        best_index = ops.nms(cnts, scrs, 0.1)[0]
        x1, y1, x2, y2 = int(cnts[best_index][0] * (w / SIZE)), \
                         int(cnts[best_index][1] * (h / SIZE)), \
                         int(cnts[best_index][2] * (w / SIZE)), \
                         int(cnts[best_index][3] * (h / SIZE))
    return x1, y1, x2, y2

# ******************************************************************************************************************** #

def get_box_coordinates_make_data(model, transform, image):
    x1, y1, x2, y2 = None, None, None, None
    temp_image = image.copy()
    with torch.no_grad():
        output = model(transform(temp_image).to(DEVICE).unsqueeze(dim=0))[0]
    cnts, scrs = output["boxes"], output["scores"]
    if len(cnts) != 0:
        cnts = ops.clip_boxes_to_image(cnts, (SIZE, SIZE))
        best_index = ops.nms(cnts, scrs, 0.1)[0]
        x1, y1, x2, y2 = int(cnts[best_index][0]), \
                         int(cnts[best_index][1]), \
                         int(cnts[best_index][2]), \
                         int(cnts[best_index][3])
    return x1, y1, x2, y2

# ******************************************************************************************************************** #

def process(image, x1, y1, x2, y2):
    if x1 is None:
        cv2.putText(img=image, text=" --- No Objects Detected ---", org=(50, 50),
                    fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), thickness=2)
    else:
        cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 255, 255), thickness=2)
    return image

# ******************************************************************************************************************** #

def normalize(x):
    for i in range(x.shape[0]):
        x[i] = (x[i] - torch.min(x[i])) / (torch.max(x[i]) - torch.min(x[i]))
    return x

# ******************************************************************************************************************** #

def get_single_image_features(model=None, transform=None, image=None):
    with torch.no_grad():
        features = model(transform(image).to(DEVICE).unsqueeze(dim=0))
    return normalize(features).detach().cpu().numpy()

# ******************************************************************************************************************** #