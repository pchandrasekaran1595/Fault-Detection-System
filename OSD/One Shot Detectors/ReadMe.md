Performs One Shot Detection by comparing current frame with the (anchor) snapshot image using TripletLoss or Cosine Similarity

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