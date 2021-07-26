import os
import cv2
import shutil
import platform
import utils as u

# Setting Up Global Variables fro ROI Extraction
x_min, y_min, x_max, y_max, cropping = 0, 0, 0, 0, False
orig_image, img_count = None, 1

# ******************************************************************************************************************** #

# Mouse Callback to handle extracting ROI from the frame
def crop_img(event, x, y, flags, param):

    global x_min, y_min, x_max, y_max, cropping, img_count

    # Start recording history of points once Left Button is Pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        x_min, y_min, x_max, y_max = x, y, x, y
        cropping = True

    # REcord and Update the last known location of the Mouse within the window
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            x_max, y_max = x, y

    # Stop recording history; save the cropped image and the original image. Minimum patch Size (32x32)
    elif event == cv2.EVENT_LBUTTONUP:
        x_max, y_max = x, y
        cropping = False

        refPoint = [(x_min, y_min), (x_max, y_max)]
        file = open(os.path.join(u.event_handler_path, "Box Reference.txt"), "w")

        if len(refPoint) == 2:
            roi = orig_image[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            if abs(refPoint[1][0] - refPoint[0][0]) >= u.MIN_CROP_WIDTH and abs(refPoint[1][1] - refPoint[0][1]) >= u.MIN_CROP_HEIGHT:
                cv2.imwrite(os.path.join(u.event_handler_path, "Snapshot_{}.png".format(img_count)), orig_image)
                cv2.imwrite(os.path.join(u.event_handler_path, "Crop_{}.png".format(img_count)), roi)
                cv2.imshow("Cropped ROI", roi)
                file.write(repr(x_min) + "," + repr(y_min) + "," + repr(x_max) + "," + repr(y_max) + ",")
                img_count += 1
                x_min, x_max, y_min, y_max = 0, 0, 0, 0
            else:
                u.myprint("PATCH TOO SMALL !!!!", "red")

        file.close()


# ******************************************************************************************************************** #

# Function that handles capture the snapshot
def capture_snapshot(part_name=None):
    u.event_handler_path = os.path.join(u.IMAGE_PATH, part_name)

    # Directory Handling
    if not os.path.exists(u.event_handler_path):
        os.makedirs(u.event_handler_path)
    else:
        shutil.rmtree(u.event_handler_path)
        os.makedirs(u.event_handler_path)

    name = "Feed"
    global orig_image, img_count, x_min, y_min, x_max, y_max

    # Setting up capture object
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(u.ID)
    else:
        cap = cv2.VideoCapture(u.ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, u.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, u.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, u.FPS)

    # Adding a name to the display Window
    cv2.namedWindow(name)

    # Binding Mouse Callback to Window
    cv2.setMouseCallback(name, crop_img)

    # Read data from capture object
    while cap.isOpened():
        _, frame = cap.read()
        orig_image = frame.copy()

        # Draw a rectangle as user draws
        cv2.rectangle(img=frame, pt1=(x_min, y_min), pt2=(x_max, y_max), color=(0, 0, 255), thickness=2)
        cv2.imshow(name, frame)

        # Press 'r' to Reset the process
        if cv2.waitKey(u.WAIT_DELAY) == ord("r"):
            x_min, x_max, y_min, y_max = 0, 0, 0, 0
            img_count -= img_count
            cv2.destroyWindow("Cropped ROI")

        # Press 'q' to Quit
        if cv2.waitKey(u.WAIT_DELAY) == ord("q"):
            img_count = 1
            break

    # Release capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

# ******************************************************************************************************************** #
