Performs One Shot Detection by comparing current frame with the (anchor) snapshot image. Uses an object detector to extract ROI from snapshot and current frame to be used in comparison.

&nbsp;

---

&nbsp;


## **CLI Arguments**

<pre>
1. --capture   : Flag that controls entry into capture mode

2. --name      : Name of the image file

3. --cliplimit : Cliplimit used in CLAHE preprocessing 
</pre>

&nbsp;

---

## **Python Scripts Information**

1. *cli.py* - Command Line Interface Implementation of the Application
2. *Detector.py* - Contains the Pytorch Deep Feature Extraction Model togerther with performing the Extraction and Detection.
3. *Snapshot.py* - Handles the capture of a frame from the webcam feed.
4. *utils.py* - Contains Constants and Utility Functions used throughout the Application.
5. *main.py* - Entry Point into the Application.

&nbsp;

---