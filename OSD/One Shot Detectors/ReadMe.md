- Performs One Shot Detection by comparing current frame with the (anchor) snapshot image using TripletLoss or Cosine Similarity

- Saves the captured frame at ./Images/Snapshot_{}.png

&nbsp;

---

&nbsp;

## **CLI Arguments**


<pre>
1. --capture   : Runs the application in capture mode. Functionality can be found in Snapshot.py

2. --triplet   : Flag that controls entry into Triplet Detection

3. --cosine    : Flag that controls entry into Cosine Similarity Detection

4. --margin    : Margin to be used with triplet loss (Default: 1.0)

5. --name      : File name of the image file

6. --clipLimit : cliplimit used in CLAHE preprocessing

Run in capture mode during first run.
</pre>

&nbsp;

---

&nbsp;

## **Python Scripts Information**

1. *cli.py* - Command Line Interface Implementation of the Application
2. *Detector.py* - Contains the Pytorch Deep Feature Extraction Model togerther with performing the Extraction and Detection.
3. *Snapshot.py* - Handles the capture of a frame from the webcam feed.
4. *utils.py* - Contains Constants and Utility Functions used throughout the Application.
5. *main.py* - Entry Point into the Application.

&nbsp;

---