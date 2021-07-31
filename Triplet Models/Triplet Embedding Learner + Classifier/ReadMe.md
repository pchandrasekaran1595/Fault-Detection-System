- Uses a Triplet of Inputs (Anchor, Positive, Negative) to learn new embeddings from the extracted features.

- Classifier (Similarity Score Predictor) takes the learned embeddings (from the embedding net in the previous step) and classifies it.

- GUI version has not been developed.

TEST : Create Dataset that uses all the Positive Directory Base Images as Anchors

&nbsp;

---

&nbsp;

## **CLI Arguments**

<pre>
1. --num-samples - Number of Samples to be created for each class in the Dataset

2. --embed       - Size of the new Embeddings

3. --e-epochs    - Number opf training epochs for the embedding net

4. --c-epochs    - Number opf training epochs for the classifying net

5. --lower       - Lower Confidence Bound of the System

6. --upper       - Lower Confidence Bound of the System

7. --id          - Device ID of the capture object
</pre>

&nbsp;

---
