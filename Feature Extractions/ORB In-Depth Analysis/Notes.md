1. Without any image preprocessing, topK Keypoints have very slight variation (flickering) when object is still.

2. Using Gaussian Blur with *(K, S) = (15, 0)* improves the results

3. *CLAHE(5, 2)* + *G(15, 0)* causes too much variation.

4. Only *CLAHE(5, 2)* provides very good results