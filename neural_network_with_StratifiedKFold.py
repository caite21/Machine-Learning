import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from itertools import product

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras import layers, models
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential


# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# stratified 5-fold cross-validation design
skf = StratifiedKFold(n_splits=5)
# Exploration parameters
filters = [16, 32]
learning_rates = [0.001, 0.01]
results = []
param_grid_2D = list(product(filters, learning_rates))

# Source: University of Alberta ----------------------
class KerasClassifier(BaseEstimator, TransformerMixin):
    def __init__(self, model_func, **kwargs):
        self.model_func = model_func
        self.kwargs = kwargs
        self.model_ = None

    def fit(self, X, y):
        self.model_ = self.model_func(**self.kwargs)
        self.model_.fit(X, y, epochs=5, verbose=0)
        return self

    def predict(self, X):
      # The argmax here converts the one-hot encoding to label format
        return np.argmax(self.model_.predict(X), axis=1)

# Create neural network model
def keras_seq_model(filters, learning_rate):
    model = Sequential()
    # Define filters and convolutional layers here
    model.add(Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    # Add a maxpooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten the output and give it to a fully connected layer
    model.add(Flatten())
    # One hidden layer maps the flattened neurons to output
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
# ------------------------------------------------


# Loop through every combination of parameters to explore
for filters, learning_rate in param_grid_2D:
    fold_scores = []
    for train_index, test_index in skf.split(X_train, y_train):
        # Cross validation splitting train data into new train and test data
        X_train_explore = X_train[train_index]
        X_test_explore = X_train[test_index]
        y_train_explore = y_train_encoded[train_index]
        y_test_explore = y_train_encoded[test_index]

        # New pipeline
        pipeline = Pipeline([('classifier', KerasClassifier(keras_seq_model,
                                                            filters=filters,
                                                            learning_rate=learning_rate))])
        pipeline.fit(X_train_explore, y_train_explore)
        # Outputs decoded y_pred
        y_pred_explore = pipeline.predict(X_test_explore)
        # Decode y_test
        y_test_explore_norm = np.argmax(y_test_explore, axis=1)
        f1 = f1_score(y_test_explore_norm, y_pred_explore, average='macro')
        fold_scores.append(f1)
    # Average F1 score
    mean_f1 = np.mean(fold_scores)
    results.append((filters, learning_rate, mean_f1))
