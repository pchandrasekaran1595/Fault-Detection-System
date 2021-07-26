"""
    Script to track keypoints
"""

import cv2
import platform

import utils as u
from Analysis import get_orb_features, get_topK

# ******************************************************************************************************************** #

def basic_tracker(nfeatures, K=5):
    prev = [int(u.CAM_WIDTH/2), int(u.CAM_HEIGHT/2)]

    # Create ORB object
    orb = cv2.ORB_create(nfeatures=nfeatures)
    
    # Setting up capture object
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(u.ID)
    else:
        cap = cv2.VideoCapture(u.ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, u.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, u.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, u.FPS)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(2, 2))

    # Frame Counter
    count = 0

    # Read data from capture object
    while cap.isOpened():
        _, frame = cap.read()
        disp_frame = frame.copy()

        for i in range(frame.shape[-1]):
            frame[:, :, i] = clahe.apply(frame[:, :, i])
        kps, _ = get_orb_features(orb, frame)
        kps = get_topK(kps, K=K)

        # Do this only if keypoints are present in the image
        if kps is not None:
            kp = kps[0]
            disp_frame = cv2.drawKeypoints(disp_frame, kps, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            
            # Update history every 15 frames
            if count == 0:
                current = [kp.pt[0], kp.pt[1]]
                buffer = [kp.pt[0], kp.pt[1]]
            elif count % 15 == 0:
                buffer = current
                prev = buffer
                current = [kp.pt[0], kp.pt[1]]
            count += 1

            # Draw a line showing the history of the keypoints
            cv2.line(img=disp_frame, pt1=(int(current[0]), int(current[1])), pt2=(int(prev[0]), int(prev[1])), color=(0, 255, 0), thickness=4)

        # Display the frame
        cv2.imshow("Tracking Feed", disp_frame)

        # Press 'q' to Quit
        if cv2.waitKey(1) == ord("q"):
            break
    
    # Release capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

# ******************************************************************************************************************** #
