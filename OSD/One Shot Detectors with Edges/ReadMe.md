Performs One Shot Detection by comparing current frame with the (anchor) snapshot image using Cosine Similarity. Current Frame and Image snapshot are both preprocessed with edge detection using pixel gradients.

&nbsp;

---

&nbsp;


## **CLI Arguments:**

<pre>
1. --capture   : Flag that controls entry into capture mode

2. --name      : Name of the image file
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