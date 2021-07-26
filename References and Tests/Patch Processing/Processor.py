"""
    Script that handles processing video as patches
"""


import cv2
import platform
import numpy as np

import utils as u

# ******************************************************************************************************************** #

def process_video_as_patches(pw, ph, test=None): 
    """
        1. --pw   : Width of the patch (in pixels) (Can be specified via command line)
        2. --pw   : Height of the patch (in pixels) (Can be specified via command line)
        3. --test : number indicating what test is to be performed (Can be specified via command line)
    """   

    # Initialize the capture object
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(u.ID)
    else:
        cap = cv2.VideoCapture(u.ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, u.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, u.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FPS, u.FPS)

    print("")

    # Read data from capture object
    while cap.isOpened():
        _, frame = cap.read()

        h, w, _ = frame.shape
        patches = []

        if h % ph == 0:num_cols = int(h/ph)
        else:num_cols = int(h/ph) + 1

        if w % pw == 0:num_rows = int(w/pw)
        else:num_rows = int(w/pw) + 1

        for i in range(0, h, ph):
            for j in range(0, w, pw):
                patches.append([i, ph+i, j, pw+j])
        patches = np.array(patches).reshape(num_cols, num_rows, 4)

        frame = cv2.resize(src=frame, dsize=(num_rows*pw, num_cols*ph), interpolation=cv2.INTER_AREA)
        new_frame = (np.ones((num_cols*ph, num_rows*pw, 3)) * 75).astype("uint8")

        for i in range(num_cols):
            for j in range(num_rows):
                if test is None:
                    new_frame[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :] = frame[patches[i][j][0]:patches[i][j][1], patches[i][j][2]:patches[i][j][3], :]

                # Test 1
                if test == 1:
                    if i % 2 == 0:
                        new_frame[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :] = frame[patches[i][j][0]:patches[i][j][1], patches[i][j][2]:patches[i][j][3], :]
                
                # Test 2
                if test == 2:
                    if j % 2 == 0:
                        new_frame[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :] = frame[patches[i][j][0]:patches[i][j][1], patches[i][j][2]:patches[i][j][3], :]
            
                # Test 3
                if test == 3:
                        if i % 2 == 0 and j % 2 == 0:
                            new_frame[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :] = frame[patches[i][j][0]:patches[i][j][1], patches[i][j][2]:patches[i][j][3], :]

        # Stack Original and Patch Processed Frame
        frame = np.hstack((frame, new_frame))

        # Display
        cv2.imshow("Patch - Original Frame - Processed Frame", frame)

        # Press 'q' to Quit
        if cv2.waitKey(u.WAIT_DELAY) == ord("q"):
            break
    
    # Release the capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    
# ******************************************************************************************************************** #