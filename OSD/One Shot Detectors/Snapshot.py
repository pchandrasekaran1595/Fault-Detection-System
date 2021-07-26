import os
import cv2
import platform

import utils as u

# ******************************************************************************************************************** #

def capture_snapshot():

    # Setting up capture object
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(u.ID)
    else:
        cap = cv2.VideoCapture(u.ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, u.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, u.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, u.FPS)
    
    count = len(os.listdir(u.IMAGE_PATH)) + 1
    u.breaker()

    # Read data from capture object
    while cap.isOpened():
        _, frame = cap.read()

        # Preprocess frame with CLAHE (clipLimit:2, tileGridSize: (2, 2))
        frame = u.clahe_equ(frame)

        # Display the frame
        cv2.imshow("Feed", frame)

        # Press 'c' to Capture the frame
        if cv2.waitKey(1) == ord("c"):
            cv2.imwrite(os.path.join(u.IMAGE_PATH, "Snapshot_{}.png".format(count)), frame)
            print("Captured Snapshot - {}".format(count))
            count += 1

        # Press 'q' to Quit
        if cv2.waitKey(1) == ord("q"):
            break
    u.breaker()
    
    # Release the capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

# ******************************************************************************************************************** #
