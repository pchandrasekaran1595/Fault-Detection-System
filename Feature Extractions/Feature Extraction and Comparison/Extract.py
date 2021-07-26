"""
    Script that handles Feature Extraction and Comparison
"""

import cv2
import platform
import numpy as np

import utils as u
from Models import build_model

# Function that handles ORB Feature comparison between Video Feed and Image File.
def ml_compare(image, nfeatures, distance=32):
    """
        image     : (np.ndarray) Image data to which comparison is to be made.
        nfeatures : (int) Number of features to initialize the ORB object with. 
        distance  : (float) Hamming Distance for Brute Force Matching. (Set via the command line)
    """

    # Setting up the capture object
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(u.ID)
    else:
        cap = cv2.VideoCapture(u.ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, u.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, u.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, u.FPS)

    # Setting up Brute Force matcher object
    BF = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)

    # Setting up ORB object
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # Obtain the keypoints and descriptors for the image file
    kp1, des1 = orb.detectAndCompute(image, None)

    # Read data from capture object
    while cap.isOpened():
        _, frame = cap.read()

        # Preprocess frame with CLAHE (clipLimit:2, tileGridSize: (2, 2))
        frame = u.clahe_equ(frame)

        # Obtain the keypoints and descriptors for the current frame
        kp2, des2 = orb.detectAndCompute(frame, None)

        # Perform Brute Force Matching (Modify this to use knnMatch)
        matches = BF.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        matches = [match for match in matches if match.distance < distance]

        # Horizontally stack the image with the frame
        frame = np.hstack((image, frame))

        # Display 'Match' or 'No Match' based on the number of matches
        if len(matches) > 5:
            cv2.putText(img=frame, text="Match", org=(25, 75), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(0, 255, 0), thickness=2)
        else:
            cv2.putText(img=frame, text="No Match", org=(25, 75), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(0, 0, 255), thickness=2)

        # Display the frame
        cv2.imshow("Feed", frame)
        
        # Press 'q' to Quit
        if cv2.waitKey(1) == ord("q"):
            break
    
    # Release capture cbject and destory all windows
    cap.release()
    cv2.destroyAllWindows()

# ******************************************************************************************************************** #

# Function that handles Deep Learning Feature comparison between Video Feed and Image File.
def dl_compare(image, similarity):
    """
        image      : (np.ndarray) Image data to which comparison is to be made. 
        similarity : (float) Cosine Similarity Threshold [0, 1]. (Set via the command line)
    """

    # Setting up the capture object
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(u.ID)
    else:
        cap = cv2.VideoCapture(u.ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, u.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, u.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, u.FPS)

    # Get the model and criterion used for comparison
    model, criterion = build_model()

    # Obtain the features from the image file
    features_1 = model.get_features(cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB))

    # Read data from capture object
    while cap.isOpened():
        _, frame = cap.read()

        # Preprocess frame with CLAHE (clipLimit:2, tileGridSize: (2, 2))
        frame = u.clahe_equ(frame)
        disp_frame = frame.copy()

        # Obtain the features from the current frame
        features_2 = model.get_features(frame)

        # Compute the cosine similarity betwwen the Feature Vectors obtained        
        cos_sim = criterion(features_1, features_2).item()

        # Display 'Match' or 'No Match' based on the cosine similarity threshold
        if cos_sim > similarity:
            cv2.putText(img=disp_frame, text="Match", org=(25, 75), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(0, 255, 0), thickness=2)
        else:
            cv2.putText(img=disp_frame, text="No Match", org=(25, 75), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(0, 0, 255), thickness=2)
        
        # Display the frame
        cv2.imshow("Feed", disp_frame)
        
        # Press 'q' to Quit
        if cv2.waitKey(1) == ord("q"):
            break
    
    # Release capture cbject and destory all windows
    cap.release()
    cv2.destroyAllWindows()

# ******************************************************************************************************************** #
