# Machine Learning Models Portfolio

Welcome to the Machine Learning Models Portfolio repository! This repository is a collection of diverse machine learning projects, each showcasing a unique application of algorithms and methodologies. The projects demonstrate both foundational and advanced machine learning techniques implemented in Python.

## Repository Overview
  - Convolutional Neural Network (CNN): [Convolutional Neural Network for CIFAR-10 Dataset](https://github.com/caite21/Machine-Learning/tree/main?tab=readme-ov-file#convolutional-neural-network-for-cifar-10-dataset)
  - Linear Regression and Logistic Regression: [Housing Price Regression Model](https://github.com/caite21/Machine-Learning/tree/main?tab=readme-ov-file#housing-price-regression-model)
  - Decision Trees and Random Forest: [Poisonous Mushrooms Decision Trees](https://github.com/caite21/Machine-Learning/tree/main?tab=readme-ov-file#poisonous-mushrooms-decision-trees)
  - Neural Network: [Wine_MLP](https://github.com/caite21/Machine-Learning/blob/main/MLP_with_backprop.py)
  - K-Nearest Neighbors (kNN): [kNN_Cancer_Classifier](https://github.com/caite21/Machine-Learning/blob/main/kNN_cancer_classifier.py)
  - Fuzzy Logic: [Tip_Decision_Fuzzy_Logic](https://github.com/caite21/Machine-Learning/blob/main/tip_decision_fuzzy_logic.py)

For another CNN for meteor detection and working with OpenCV, see [Autonomous-Meteor-Detection](https://github.com/caite21/Autonomous-Meteor-Detection)


## Convolutional Neural Network for CIFAR-10 Dataset

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



## Housing Price Regression Model

[📁 House_Price_Regression](https://github.com/caite21/Machine-Learning/blob/main/House_Price_Regression.ipynb)

This project analyzes a housing dataset by preprocessing categorical data with one-hot encoding, dropping uncorrelated features, and fitting a linear regression model that achieves an R-squared value of 0.86. It further explores price classification by creating two categories: above or below $175,000, and applies a logistic regression model, reaching an F1 score of 0.93.

![price_linear_regression](https://github.com/user-attachments/assets/9d96bcc5-4d50-4b32-8048-4cd3a13d863b)

Scatter plot comparing actual vs. predicted house prices from the linear regression model. The red dashed line represents a perfect prediction (where predicted prices equal actual prices), highlighting the model's performance in predicting house prices with an R-squared value of 0.86.

![price_logistic_regression](https://github.com/user-attachments/assets/64ea060b-7924-4a89-b9b5-8aa809878fac)

Visualizes the distribution of house prices in the dataset and illustrates how the logistic regression model classifies prices as either above or below $175,000.

![price_logistic_matrix](https://github.com/user-attachments/assets/2b838307-c458-414a-a820-bebca9705f3d)

Confusion matrix showing the logistic regression model's performance in classifying house prices as above $175,000 (1) or below $175,000 (0), with an F1 score of 0.93.


## Poisonous Mushrooms Decision Trees 

[📁 Poisonous_Mushroom_Decision_Trees](https://github.com/caite21/Machine-Learning/blob/main/Poisonous_Mushroom_Decision_Trees.ipynb)

This Jupyter Notebook compares three decision tree models—Gini index, Entropy, and Random Forest—to classify mushrooms as edible or poisonous. Each model is tuned for optimal hyperparameters, achieving 100% accuracy. Cross-validation results are also included to assess model performance.

![mushroom_decision_tree](https://github.com/user-attachments/assets/fcdeac78-6f5a-49d8-80d8-a099c6f5d437)

Decision tree visualization for classifying mushrooms as poisonous or edible based on their characteristics, using entropy as the splitting criterion and achieving 100% accuracy.




## Acknowledgments
- [Scikit-learn](https://scikit-learn.org/stable/) for providing machine learning tools.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for deep learning frameworks.
- [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/doc/stable/) for data manipulation and analysis.
