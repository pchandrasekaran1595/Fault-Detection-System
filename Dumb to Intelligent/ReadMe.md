- Uses a VGG16 (with Batch Normalization) as the backbone pretrained network

- This is the main application

- Assumes that the user can provide only a single positive sample for a single component. System must be subsequently trained for each new component added, or if and when more samples become available for a pre-existing component

- Does not contain a CLI version

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