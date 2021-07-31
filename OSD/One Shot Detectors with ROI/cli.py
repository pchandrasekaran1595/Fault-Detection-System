"""
    CLI Application
"""

import os
import cv2
import sys

import utils as u
from Snapshot import capture_snapshot
from Detector import CosineDetector

# ******************************************************************************************************************** #

def app():
    args_1 = "--capture"
    args_2 = "--name"
    args_3 = "--cliplimit"

    do_capture = None, None
    name = "Snapshot_1.png"
    clipLimit = 2.0

    if args_1 in sys.argv:
        do_capture = True
    if args_2 in sys.argv:
        name = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv:
        clipLimit = float(sys.argv[sys.argv.index(args_3) + 1])
    
    if do_capture:
        capture_snapshot(clipLimit)
    else:
        image = u.preprocess(cv2.imread(os.path.join(u.IMAGE_PATH, name), cv2.IMREAD_COLOR))
        CosineDetector(image, clipLimit)

# ******************************************************************************************************************** #
