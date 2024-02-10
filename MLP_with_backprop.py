# suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer

from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from itertools import product


file_name = "wine.data"
df = pd.read_csv(file_name, header=None)
header = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
          'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
          'Proanthocyanins', 'Color intensity', 'Hue',
          'OD280/OD315 of diluted wines', 'Proline']
num_attr = 13
df.columns = header

# separate labels and attributes
y = df['class']
X = df.drop('class', axis=1)

# scale
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# mean imputation
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# one-hot encode classes
encoder = OneHotEncoder(sparse=False)
y_reshaped = y.values.reshape(-1, 1)
y = encoder.fit_transform(y_reshaped)

# split dataset into a training and testing set
test_size = 0.10
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

# exploration parameters
layers = [1, 2, 3]
neurons = [32, 64, 128]
learning_rate = [0.001, 0.01, 0.1]
momentum = 0.9
results = []
param_grid_2D = list(product(layers, neurons, learning_rate))

# tenfold cross-validation design
skf = StratifiedKFold(n_splits=10)

# create neural network model
def keras_seq_model(layers, neurons, learning_rate):
    model = keras.Sequential()
    # add input layers
    for i in range(layers):
        model.add(keras.layers.Dense(neurons, activation='relu'))
    # output layer for 3 classes
    model.add(keras.layers.Dense(3, activation='softmax'))
    # compile with backpropagation and momentum
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=momentum),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# loop through every combination of parameters to explore
for layers, neurons, learning_rate in param_grid_2D:
    fold_scores = []
    for train_index, test_index in skf.split(X, y.argmax(1)):
        X_train_explore = X[train_index]
        X_test_explore = X[test_index]
        y_df = pd.DataFrame(y)
        y_train_explore = y_df.iloc[train_index]
        y_test_explore = y_df.iloc[test_index]

        model = keras_seq_model(layers, neurons, learning_rate)
        model.fit(X_train_explore, y_train_explore)

        y_pred = model.predict(X_test_explore)
        y_pred_binary = (y_pred == y_pred.max(axis=1)[:, None]).astype(int)
        f1 = f1_score(y_test_explore, y_pred_binary, average='samples')
        fold_scores.append(f1)

    mean_f1 = np.mean(fold_scores)
    results.append((layers, neurons, learning_rate, mean_f1))


# print best parameters
best_result = max(results, key=lambda x: x[3])
print(f"Best parameters:")
print(f"Layers = {best_result[0]}")
print(f"Neurons = {best_result[1]}")
print(f"Learning Rate = {best_result[2]}")
print(f"Best F1 Score: {best_result[3]:.4f}\n")

# train using best parameter settings
best_model = keras_seq_model(best_result[0], best_result[1], best_result[2])
best_model.fit(X_train, y_train)
# evaluate on test data
y_pred = best_model.predict(X_test)
y_pred_binary = (y_pred == y_pred.max(axis=1)[:, None]).astype(int)
f1 = f1_score(y_test, y_pred_binary, average='samples')
print(f"\nTest F1 Score: {f1:.4f}")
