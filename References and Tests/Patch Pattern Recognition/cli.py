"""
    CLI Arguments:
        1. --part-name  : Component/Part Name
        2. --filename   : Filename of the image file (Used only during --process)
        3. --capture    : Flag that controls entry into capture mode
        4. --process    : Flag that controls entry into process mode
        5. --similarity : Cosine Similarity Threshold (Default: 0.8) 
"""

import os
import cv2
import sys

import utils as u
from Snapshot import capture_snapshot
from Processor import process_patches_in_video

# ******************************************************************************************************************** #

def app():
    args_1 = "--part-name"
    args_2 = "--filename"
    args_3 = "--capture"
    args_4 = "--process"
    args_5 = "--similarity"

    # Default CLI Argument Values
    do_capture, do_process = None, None
    similarity = 0.8

    # CLI Argument Handling
    if args_1 in sys.argv:
        p_name = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv:
        f_name = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv:
        do_capture = True
    if args_4 in sys.argv:
        do_process = True
    if args_5 in sys.argv:
        similarity = float(sys.argv[sys.argv.index(args_5) + 1])
    
    u.breaker()
    u.myprint("--- Application Start ---", color="green")

    # Runs if --capture is specified
    if do_capture:
        capture_snapshot(part_name=p_name)

    # Runs if --process is specified
    if do_process:
        path = os.path.join(u.IMAGE_PATH, p_name)
        patch = u.preprocess(cv2.imread(os.path.join(path, f_name), cv2.IMREAD_COLOR))
        process_patches_in_video(patch, similarity)

    u.myprint("\n--- Application End ---", color="green")
    u.breaker()

# ******************************************************************************************************************** #
