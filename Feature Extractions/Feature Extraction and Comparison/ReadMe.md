- Application that extracts features from a captured image and compares it with the realtime webcam feed using the Cosine Similarity Metric

&nbsp;

---

&nbsp;

## **CLI Arguments**

<pre>
1. --capture    : Runs the application in capture mode. Functionality can be found in Snapshot.py
2. --ml         : Flag that controls entry into Machine Learning mode (ORB)
3. --features   : Number of features to be used with ORB (Default: 500)
4. --distance   : Hamming Distance used to compare ORB Features (Default: 32)
5. --dl         : Flag that controls entry into Deep Learning mode (VGG26)
6. --similarity : Cosine Similarity Threshold (Default: 0.85)
7. --filename   : Name of the file to compare the Realtime Video feed to.
</pre>

&nbsp;

---