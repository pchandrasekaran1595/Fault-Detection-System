import os
import cv2
import platform
import numpy as np
import random as r
import matplotlib.pyplot as plt

import utils as u

#####################################################################################################

# Function to get ORB Features from an image.
def get_orb_features(orb, image):
    """
        orb: ORB Object
        image: (np.ndarray) image data
    """
    image = cv2.cvtColor(src=image, code=cv2.COLOR_RGB2GRAY)
    kp, des = orb.detectAndCompute(image, None)
    return kp, des


# Fucntion to return topK keypoints. TopK keypoints are decided by their response
def get_topK(kp, K=5):
    """
        kp: List of ORB Keypoints
        K: Top K keypoints to use (Default: 5)
    """
    responses = []
    for k in kp:
        responses.append(k.response)
    
    if len(responses) > K:
        responses_idx = np.argpartition(responses, -K)[-K:].astype("int64")
        topK_kp = []
        for idx in responses_idx:
            topK_kp.append(kp[idx])
        return topK_kp
    else:
        return None


# Image Analysis tool showing various keypoint attributes for a random keypoint
def show_kp_info(kp):
    kp_sample = r.randint(0, len(kp)-1)

    u.breaker()
    print("Total Number of Keypoints : {}".format(len(kp)))
    u.breaker()
    print("Keypoint {} Information".format(kp_sample))
    u.breaker()

    print("Angle     : {:.5f}".format(kp[kp_sample].angle))
    print("Octave    : {:.5f}".format(kp[kp_sample].octave))
    print("Point     : {}".format(kp[kp_sample].pt))
    print("Response  : {:.5f}".format(kp[kp_sample].response))
    print("Size      : {:.5f}".format(kp[kp_sample].size))


# Image Analysis tool showing various keypoint attributes the entire list of keypoints
def show_info(kp):
    angles = []
    octaves = []
    points = []
    responses = []
    sizes = []

    for k in kp:
        angles.append(k.angle)
        octaves.append(k.octave)
        points.append(k.pt)
        responses.append(k.response)
        sizes.append(k.size)

    x_Axis = np.arange(1, len(kp)+1)

    plt.figure()
    plt.subplot(1, 4, 1)
    plt.plot(x_Axis, angles, "r")
    plt.grid()
    plt.title("Angles")
    plt.subplot(1, 4, 2)
    plt.plot(x_Axis, octaves, "r")
    plt.grid()
    plt.title("Octaves")
    plt.subplot(1, 4, 3)
    plt.plot(x_Axis, responses, "r")
    plt.grid()
    plt.title("Responses")
    plt.subplot(1, 4, 4)
    plt.plot(x_Axis, sizes, "r")
    plt.grid()
    plt.title("Sizes")
    plt.show()


# Display the Keypoint image
def show_kp_image(image, kp):
    """
        image: (np.ndarray) Image data
        kp: (list) List of ORB Keypoints
    """
    kp_image = cv2.drawKeypoints(image, kp, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    plt.figure()
    plt.imshow(kp_image)
    plt.axis("off")
    plt.show()

#####################################################################################################

# Function to handle image analysis
def image_analysis(name, nfeatures, K):
    
    # Read Image file
    image = cv2.cvtColor(src=cv2.imread(os.path.join(u.IMAGE_PATH, name)), code=cv2.COLOR_BGR2RGB)
    
    # Create ORB object
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # Obtain image keypoints
    kp, _ = get_orb_features(orb, image)

    # Show Random Keypoint Information
    show_kp_info(kp)

    # Show all keypoint information
    show_info(kp)
    
    # Get topK Keypoints
    topK_kp = get_topK(kp, K)

    # Show image with all the keypoints
    show_kp_image(image, kp)

    # Show image with topK keypoints
    show_kp_image(image, topK_kp)

#####################################################################################################

def realtime_analysis(nfeatures, K=None):
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

    # Creat CLAHE object
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(2, 2))
    while cap.isOpened():
        _, frame = cap.read()

        # Preprocess frame with CLAHE (clipLimit:2, tileGridSize: (2, 2)) 
        for i in range(frame.shape[-1]):
            frame[:, :, i] = clahe.apply(frame[:, :, i])
        # frame = cv2.GaussianBlur(src=frame, ksize=(15, 15), sigmaX=0)

        # Obtain frame keypoints
        kp, _ = get_orb_features(orb, frame)

        # Get topK Keypoints if K is specified
        if K is None:
            frame = cv2.drawKeypoints(frame, kp, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        else:
            topK_kp = get_topK(kp, K=K)
            if topK_kp is not None:
                frame = cv2.drawKeypoints(frame, topK_kp, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # Press 'q' to Quit
        cv2.imshow("Feed", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    
    # Release capture cbject and destory all windows
    cap.release()
    cv2.destroyAllWindows()

#####################################################################################################
