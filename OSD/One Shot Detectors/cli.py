"""
    CLI Application
"""

import os
import cv2
import sys

import utils as u
from Snapshot import capture_snapshot
from Detector import TripletDetector, CosineDetector

# ******************************************************************************************************************** #

def app():
    args_1 = "--capture"
    args_2 = "--triplet"
    args_3 = "--cosine"
    args_4 = "--margin"
    args_5 = "--name"
    args_6 = "--cliplimit"

    # Default CLI Argument Values
    do_capture, do_triplet, do_cosine = None, None, None
    margin = 1.0
    name = "Snapshot_1.png"
    clipLimit = 2.0

    # CLI Argument Handling
    if args_1 in sys.argv:
        do_capture = True
    if args_2 in sys.argv:
        do_triplet = True
    if args_3 in sys.argv:
        do_cosine = True
    if args_4 in sys.argv:
        margin = float(sys.argv[sys.argv.index(args_4) + 1])
    if args_5 in sys.argv:
        name = sys.argv[sys.argv.index(args_5) + 1]
    if args_6 in sys.argv:
        clipLimit = float(sys.argv[sys.argv.index(args_6) + 1])
    
    # Runs if --capture is specified
    if do_capture:
        capture_snapshot(clipLimit)
    
    # Runs if --triplet is specified
    if do_triplet:
        try:
            image = u.preprocess(cv2.imread(os.path.join(u.IMAGE_PATH, name), cv2.IMREAD_COLOR))
            TripletDetector(image, margin, clipLimit)
        except:
            u.breaker()
            print("Possible Problem reading Image File")
            u.breaker()
        finally:
            pass
    
    # Runs if --cosine is specifier
    if do_cosine:
        try:
            image = u.preprocess(cv2.imread(os.path.join(u.IMAGE_PATH, name), cv2.IMREAD_COLOR))
            CosineDetector(image, clipLimit)
        except:
            u.breaker()
            print("Possible Problem reading Image File")
            u.breaker()
        finally:
            pass

# ******************************************************************************************************************** #
