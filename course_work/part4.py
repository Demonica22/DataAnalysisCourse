import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
import scikitplot as skplt

df = pd.read_csv('ready_data.csv')

Y = df['EngagementLevel']
X = df.drop('EngagementLevel', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=4, stratify=Y)
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# model = RandomForestClassifier(n_estimators=20, max_depth=13).fit(X_train, y_train)
# y_pred = model.predict(X_test)
# model = DecisionTreeClassifier(max_depth=11).fit(X_train, y_train)
# y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d")
print(classification_report(y_test, y_pred))
plt.title(f"Confusion Matrix")
plt.xlabel("Predicted ")
plt.ylabel("True")
plt.show()
roc_auc = roc_auc_score(y_test == 0, y_pred == 0)
print(f"{roc_auc=}")
fpr, tpr, _ = roc_curve(y_test == 0, y_pred == 0)
plt.plot(fpr, tpr)
plt.xlabel("FalsePositiveRate")
plt.ylabel("TruePositiveRate")
plt.title(f"ROC кривая для параметра 0")
plt.show()
"""
13
0.897464718371425
train accuracy  0.9675243567324506
При глубине деревьев в лесу 13 

Параметры max_leaf_nodes и min_samples_leaf заметных изменений метрик не дали.
"""
# total_accuracy = accuracy_score(y_test, y_pred)
# print(f"{total_accuracy=}")
# y_pred_proba = model.predict_proba(X_test)
# plt.subplot()
#
# for i in [0, 1, 2]:
#     print(f"Binary param: {i}")
#     y_pred_bin = y_pred == i
#     y_test_bin = y_test == i
#     accuracy = accuracy_score(y_test_bin, y_pred_bin)
#     precision = precision_score(y_test_bin, y_pred_bin)
#     recall = recall_score(y_test_bin, y_pred_bin)
#     print(f"{accuracy=}, {precision=}, {recall=}")
#     roc_auc = roc_auc_score(y_test_bin, y_pred_bin)
#     print(f"{roc_auc=}")
#     fpr, tpr, _ = roc_curve(y_test_bin, y_pred_bin)
#     plt.plot(fpr, tpr,label=f"{i}" )
#     # skplt.metrics.plot_roc_curve(y_test_bin, y_pred_proba)
# plt.figlegend()
#
# plt.xlabel("FalsePositiveRate")
# plt.ylabel("TruePositiveRate")
# plt.title(f"ROC кривая")
# plt.show()
