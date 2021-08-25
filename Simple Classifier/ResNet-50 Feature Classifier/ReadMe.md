- Extracted features from the dataset are passed through a Multi-Layered Percpetron Classifier and inference is drawn in realtime. 

- Allows for addition of False Positive and False Negative upon button presses after which models can be retrained.

- Uses a VGG16 (with Batch Normalization) as the backbone pretrained network

- This is to be used for testing

&nbsp;

---

&nbsp;

## **CLI Arguments**

<pre>
1. --num-samples - Number of Samples to be created for each class in the Dataset

2. --embed       - Size of the new Embeddings

3. --epochs      - Number of training epochs

4. --id          - Device ID of the capture device

5. --lower       - Lower Confidence Bound of the System

6. --upper       - Lower Confidence Bound of the System

7. --early       - Number of epochs to wait after stagnated validation metrics before stopping the training
</pre>

&nbsp;

---