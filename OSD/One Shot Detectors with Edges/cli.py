"""
    CLI Arguments:
        1. --capture   : Flag that controls entry into capture mode
        2. --name      : Name of the image file
"""

import os
import cv2
import sys

import utils as u
from Snapshot import capture_snapshot
from Detector import  CosineDetector


def app():
    args_1 = "--capture"
    args_2 = "--name"

    # Default CLI Arguments
    do_capture = False
    name = "Snapshot_1.png"

    # CLI Argument Handling
    if args_1 in sys.argv:
        do_capture = True
    if args_2 in sys.argv:
        name = sys.argv[sys.argv.index(args_2) + 1]
    
    if do_capture:
        capture_snapshot()
    else:
        try:
            image = u.preprocess(cv2.imread(os.path.join(u.IMAGE_PATH, name), cv2.IMREAD_COLOR))
            CosineDetector(image)
        except:
            u.breaker()
            print("Possible Problem reading Image File")
            u.breaker()
        finally:
            pass
