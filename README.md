# Machine Learning Models Portfolio

Welcome to the Machine Learning Models Portfolio repository! This repository is a collection of diverse machine learning projects, each showcasing a unique application of algorithms and methodologies. The projects demonstrate both foundational and advanced machine learning techniques implemented in Python.

## Repository Overview
- Deep Learning using TensorFlow:
  - Convolutional Neural Network (CNN): [Convolutional Neural Network for CIFAR-10 Dataset](https://github.com/caite21/Machine-Learning/edit/main/README.md#convolutional-neural-network-for-cifar-10-dataset)
  - Neural Network: [Wine_MLP](https://github.com/caite21/Machine-Learning/blob/main/MLP_with_backprop.py)

- Classical Machine Learning using Scikit-learn:
  - Linear Regression and Logistic Regression: [House_Price_Regression](https://github.com/caite21/Machine-Learning/blob/main/House_Price_Regression.ipynb)
  - Decision Trees and Random Forest: [Poisonous_Mushroom_Decision_Trees](https://github.com/caite21/Machine-Learning/blob/main/Poisonous_Mushroom_Decision_Trees.ipynb)
  - K-Nearest Neighbors (kNN): [kNN_Cancer_Classifier](https://github.com/caite21/Machine-Learning/blob/main/kNN_cancer_classifier.py)
  - Fuzzy Logic: [Tip_Decision_Fuzzy_Logic](https://github.com/caite21/Machine-Learning/blob/main/tip_decision_fuzzy_logic.py)


## Details

### Convolutional Neural Network for CIFAR-10 Dataset

[📁 CNN_CIFAR10](https://github.com/caite21/Machine-Learning/blob/main/CNN_CIFAR10.ipynb)

This project trains CNNs on the CIFAR-10 dataset (60,000 32x32 images, 10 classes) to optimize performance and enable user predictions.

**Highlights:**
- **Models Tested:** 5 variations, with 3-layer architectures showing the best generalization (up to 73% test accuracy). Dropout (50%) improves results.
- **Insights:** Pooling layers aid generalization; larger layers slightly boost accuracy. Training beyond 10 epochs has minimal impact.

**Try It:**

Download the pre-trained model (CNN_CIFAR10.keras) and use the provided notebook to predict custom 32x32 images.

**Images:**

![automobile_cnn_prediction](https://github.com/user-attachments/assets/27c2e2a8-debc-40b1-be35-4927b631f2e5)

**Example 1:**
A sample 32x32 image of an automobile was correctly classified as an automobile. The model's predictions, visualized in a bar plot, show high confidence for "automobile" with lower probabilities for "truck" and "boat," and minimal likelihood for any animal classes. This demonstrates the model's ability to distinguish vehicles from unrelated categories.


![horse_cnn_prediction](https://github.com/user-attachments/assets/b4d0764c-6e57-43fe-8677-ba466ef0ebb6)

**Example 2:**
A 32x32 image of a horse was incorrectly classified as a deer, with "horse" as the second most likely prediction. The model confidently ruled out unrelated categories like "automobile." While not perfect, the prediction was reasonably close, highlighting areas for improvement in the model's accuracy.



## Acknowledgments
- [Scikit-learn](https://scikit-learn.org/stable/) for providing machine learning tools.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for deep learning frameworks.
- [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/doc/stable/) for data manipulation and analysis.
