import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve


file_name  = ""  # removed
df = pd.read_csv(file_name)


# Correlation matrix:
corr = df.corr(numeric_only=True)
corr.style.background_gradient(cmap='coolwarm').set_precision(3)

# classification threshold
threshold = 0.75
df[prediction] = df[model_score] > threshold

print("Confusion Matrix:")
print(confusion_matrix(df[target], df[prediction]))

# Accuracy:
print("Simple accuracy=", accuracy_score(df[target], df[prediction]) )

tot_recall = recall_score(df[prediction],df[target]) + recall_score(df[target], df[prediction])
aca = 1/2*tot_recall
print("ACA=", aca)


# Cumulative Gain Chart:
sum_target_P = (df[target] == True).sum()

# aggregate for each decile, count the target positives
df[target_p_count] = np.where(df[target], 1, 0)

# sort by score
df = df.sort_values(by=model_score, ascending=False).reset_index()

# new Decile column
df['Decile'] = [i for i in range(1, 11) for _ in range(3)]

# aggregate 
aggr_df = df_1.groupby('Decile').agg({target_p_count: 'sum'}).reset_index()

# for each decile, gain = P/tot_P
aggr_df[gain] = aggr_df[target_p_count] / sum_target_P

# for each decile, cum_gain = last gain plus current gain
aggr_df[cum_gain] = aggr_df[gain].cumsum()


# plotting
plt.plot(aggr_df['Decile'], aggr_df[cum_gain], marker='o', label='Model 1')
plt.plot([0, 10], [0, 1], '--', color='gray')

plt.xlabel('Decile')
plt.ylabel('Cumulative Gain')
plt.title('Cumulative Gain Chart')
plt.legend()
plt.show()


# ROC Curve:
fpr1, tpr1, thresholds1  = roc_curve(df[target], df[model_score])

# plotting
plt.plot(fpr1, tpr1, label='Model 1')
plt.plot([0, 1], [0, 1], '--', color='gray')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()