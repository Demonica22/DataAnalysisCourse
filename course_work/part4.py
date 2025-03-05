import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

df = pd.read_csv('ready_data.csv')

Y = df['EngagementLevel']
X = df.drop('EngagementLevel', axis=1)
scaler = StandardScaler()
print(X)
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=42, stratify=Y)
# model = LogisticRegression().fit(X_train, y_train)
# Y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, Y_pred)
# print(accuracy)
for i in range(10, 300, 10):
    clf = RandomForestClassifier(n_estimators=20).fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(i)
    print(accuracy_score(y_test, predicted))
