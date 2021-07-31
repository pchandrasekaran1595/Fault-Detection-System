- Performs pattern recogniton on patches of the video frame in accordance to a reference patch selected by the user.

&nbsp;

---

&nbsp;

## **CLI Arguments:**

<pre>
1. --part-name  : Component/Part Name

2. --filename   : Filename of the image file (Used only during --process)

3. --capture    : Flag that controls entry into capture mode

4. --process    : Flag that controls entry into process mode

5. --similarity : Cosine Similarity Threshold (Default: 0.8) 
</pre>

&nbsp;

---

## **Python Scripts Information**

1. *cli.py* - Command Line Interface Implementation of the Application
2. *Models.py* - Contains the Pytorch Deep Feature Extraction Model
3. Processor.py - Contains the fucntion that split video feed into patches before processing
4. *Snapshot.py* - Handles the capture of a frame from the webcam feed.
5. *utils.py* - Contains Constants and Utility Functions used throughout the Application
6. *main.py* - Entry Point into the Application

&nbsp;

---