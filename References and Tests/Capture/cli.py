import os
import cv2
import sys
import platform
from termcolor import colored

# ******************************************************************************************************************** #

# Setting up self-aware Image Capture Directory
SAVE_PATH = os.path.join(os.getcwd(), "Captures")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Linebreaker to improve readability of output. Can be used wherever necessary.
def breaker(num=50, char="*"):
    print(colored("\n" + num*char + "\n", color="cyan"))

# Setting up capture object
def init_video(ID, w, h):
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(ID)
    else:
        cap = cv2.VideoCapture(ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)

    return cap

# ******************************************************************************************************************** #

# Function that captures an igae form the webcam feed
def image_capture(CaptureObject):
    """
        CaptureObject: OpenCV VideoCapture Object that has been initialized
    """
    count = 1
    breaker()

    # Read data from capture object
    while CaptureObject.isOpened():
        _, frame = CaptureObject.read()

        # Press 'c' to Capture frame
        if cv2.waitKey(1) == ord("c"):
            cv2.imwrite(os.path.join(SAVE_PATH, "Snapshot_{}.png".format(count)), frame)
            print("Captured Snapshot - {}".format(count))
            count += 1

        # Display the frame
        cv2.imshow("Webcam Feed", frame)

        # Press 'q' to Quit
        if cv2.waitKey(1) == ord("q"):
            break
    breaker()

    # Release the capture object and destroy all windows
    CaptureObject.release()
    cv2.destroyAllWindows()

# ******************************************************************************************************************** #

# Functionality to capture a snippet of the realtime webcam feed. Not Implemented
def video_capture(CaptureObject, number_of_frames):
    pass

# ******************************************************************************************************************** #

"""
    CLI Arguments:
        1. --id : Device ID used for Video Capture (default: 0)
        2. --w  : Width fo the Capture  Frame (Default: 640)
        3. --h  : Height of the Capture Frame (Default: 360)
"""

def app():
    args_1 = "--id"
    args_2 = "--w"
    args_3 = "--h"

    # Default CLI Argument Values
    device_id = 0
    w = 640
    h = 360

    if args_1 in sys.argv:
        device_id = int(sys.argv[sys.argv.index(args_1) + 1])
    if args_2 in sys.argv:
        w = int(sys.argv[sys.argv.index(args_2) + 1])
    if args_3 in sys.argv:
        h = int(sys.argv[sys.argv.index(args_3) + 1])
    
    cap = init_video(device_id, w, h)
    image_capture(cap)

# ******************************************************************************************************************** #
