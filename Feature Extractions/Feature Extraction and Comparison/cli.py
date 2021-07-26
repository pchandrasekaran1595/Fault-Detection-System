"""
    CLI Arguments
        1. --capture    : Runs the application in capture mode. Functionality can be found in Snapshot.py
        2. --ml         : Flag that controls entry into Machine Learning mode (ORB)
        3. --features   : Number of features to be used with ORB (Default: 500)
        4. --distance   : Hamming Distance used to compare ORB Features (Default: 32)
        5. --dl         : Flag that controls entry into Deep Learning mode (VGG26)
        6. --similarity : Cosine Similarity Threshold (Default: 0.85)
        7. --filename   : Name of the file to compare the Realtime Video feed to.
"""

import os
import sys
import cv2

import utils as u
from Snapshot import capture_snapshot
from Extract import ml_compare, dl_compare

# ******************************************************************************************************************** #

def app():
    args_1 = "--capture"

    args_2 = "--ml"
    args_3 = "--nfeatures"
    args_4 = "--distance"

    args_5 = "--dl"
    args_6 = "--similarity"

    args_7 = "--filename" 

    # Default CLI Argument Values
    do_capture, do_ml, do_dl = None, None, None
    name = "Snapshot_1.png"
    nfeatures = 500
    distance = 32
    similarity = 0.85

    # CLI Argument Handling
    if args_1 in sys.argv:
        do_capture = True

    if args_2 in sys.argv:
        do_ml = True
    if args_3 in sys.argv:
        nfeatures = int(sys.argv[sys.argv.index(args_3) + 1])
    if args_4 in sys.argv:
        distance = int(sys.argv[sys.argv.index(args_4) + 1])

    if args_5 in sys.argv:
        do_dl = True
    if args_6 in sys.argv:
        similarity = float(sys.argv[sys.argv.index(args_6) + 1])

    if args_7 in sys.argv:
        name = sys.argv[sys.argv.index(args_7) + 1]
    
    # Runs if --capture is specified
    if do_capture:
        capture_snapshot()
    
    # Runs if --ml is specified. Also needs --filename to be specified to work
    if do_ml:
        image = cv2.imread(os.path.join(u.IMAGE_PATH, name), cv2.IMREAD_COLOR)
        ml_compare(image, nfeatures, distance)
    
    # Runs if --dl is specified. Also needs --filename to be specified to work
    if do_dl:
        image = cv2.imread(os.path.join(u.IMAGE_PATH, name), cv2.IMREAD_COLOR)
        dl_compare(image, similarity)

# ******************************************************************************************************************** #
