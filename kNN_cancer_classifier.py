import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score



###### Data Injection and Preprocessing ######
file_name = "wdbc.data"
real_feature_names = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness',
                      'Compactness', 'Concavity', 'Concave Points', 'Symmetry',
                      'Fractal Dimension']
computed_feature_names = ['Mean', 'Standard Error', 'Worst']
num_feat = 10

df = pd.read_csv(file_name, header=None)

# Get rid of ID attribute
df = df.drop(0, axis=1)
# Separate labels and attributes
df_y = df.iloc[:, [0]]
df_X = df.drop(columns=df.columns[0])

# Name the attribute columns for plotting
header = []
for group_name in computed_feature_names:
  for feat_name in real_feature_names:
    header.append(group_name + " " + feat_name)
df_X.columns = header

# Plotting statistics of each attribute
num_plots = 10
diff = 10
fig, axes = plt.subplots(num_plots + 1, 1, figsize=(8, 36))
plt.subplots_adjust(hspace=0.40)

for i, ax in enumerate(axes[:10]):
    # 3 boxplots in each plot
    boxplot_cols = df_X.iloc[:, [diff*0+i, diff*1+i, diff*2+i]]

    boxplot_cols.plot(kind='box', ax=ax)
    ax.set_title(f"Plot {i + 1}: {real_feature_names[i]}")

# Mean imputation
df_X = df_X.fillna(df_X.mean())

# Transform class labels to 0 for benign and 1 for malignant
df_y = df_y.replace('B',0)
df_y = df_y.replace('M',1)

# Split your dataset into a training and testing set
X = df_X.iloc[:, :].values
y = df_y.iloc[:, 0]

test_size = 0.1
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)



###### Build AI Pipeline ######
param_grid = {'n_neighbors': [x for x in range(1, 11)]}
num_folds = 5
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', GridSearchCV(
        KNeighborsClassifier(),
        param_grid={'n_neighbors': list(range(1, 11))},
        cv=num_folds,
        scoring='f1',
        refit=True))
    ])

# Train the AI
pipeline.fit(X_train, y_train)
best_k = pipeline.named_steps['classifier'].best_params_['n_neighbors']

# Compare performances using F1 scores with a plot
k_values = pipeline.named_steps['classifier'].cv_results_['param_n_neighbors']
f1_scores = pipeline.named_steps['classifier'].cv_results_['mean_test_score']

axes[10].plot(k_values, f1_scores, marker='o')
axes[10].grid(True)
axes[10].set_ylabel("F1 Score")
axes[10].set_xlabel("k Value")
axes[10].set_title("Plot 11: F1 Score vs. k Value")



###### Test and evaluate the AI ######
y_pred = pipeline.predict(X_test)
test_f1_score = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Display results
print("###### Results ######")
print(f"Best choice of k value: {best_k}")
print()
print(f"Training Data F1 Score:\t{pipeline.named_steps['classifier'].best_score_:.4f}")
print(f"Test Data F1 Score:\t{test_f1_score:.4f}")
print(f"Test Data Recall Score:\t{recall:.4f}")
print(f"Test Data Precision:\t{precision:.4f}")
print()
print(f"Seed value:\t{seed}")
print(f"Test size: \t{test_size}")
print(f"CV folds:\t{num_folds}")


print(f"\n###### Plots ######")
plt.show()
