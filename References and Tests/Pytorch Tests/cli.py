"""
    CLI Arguments:
        1. --classify : Flag that controls entry to perform Classification
        2. --detect   : Flag that controls entry to perform Detection
        3. --segment  : Flag that controls entry to perform Segmentation
        4. --id       : Device ID of the capture device

        Needs --classify, --detect or --segment
"""

import cv2
import sys
import platform

import utils as u
from Models import Model

# ******************************************************************************************************************** #

def app():
    args_1 = "--classify"
    args_2 = "--detect"
    args_3 = "--segment"
    args_4 = "--all"
    args_5 = "--id"

    # Default CLI Argument Values
    do_classify, do_detect, do_segment, do_all = None, None, None, None

    # CLI Argument Handling
    if args_1 in sys.argv:
        do_classify = True
    if args_2 in sys.argv:
        do_detect = True
    if args_3 in sys.argv:
        do_segment = True
    if args_4 in sys.argv:
        do_all = True
    if args_5 in sys.argv:
        u.device_id = int(sys.argv[sys.argv.index(args_5) + 1])
    
    # Initialize model for classification
    if do_classify:
        model = Model(modeltype="classifier")
        model.eval()
    
    # Initialize model for detection
    if do_detect:
        model = Model(modeltype="detector")
        model.eval()
    
    # Initialize model for segmentation
    if do_segment:
        model = Model(modeltype="segmentor")
        model.eval()
    
    # Load model onto the device
    model.to(u.DEVICE)

    # Setting up capture object
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(u.device_id)
    else:
        cap = cv2.VideoCapture(u.device_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, u.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, u.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FPS, u.FPS)

    # Read data from capture object
    while cap.isOpened():
        _, frame = cap.read()

        # Runs if --classify is set
        if do_classify:
            frame = u.classify(model, frame)

        # Runs if --detect is set
        if do_detect:
            if do_all is None:
                frame = u.detect(model, frame)
            else:
                frame = u.detect_all(model, frame)
        
        # Runs if --segment is set
        if do_segment:
            frame = u.segment(model, frame)
        
        # Display the frame
        cv2.imshow("Feed", frame)

        # Press 'q' to Quit
        if cv2.waitKey(1) == ord("q"):
            break
    
    # Release the capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

# ******************************************************************************************************************** #
