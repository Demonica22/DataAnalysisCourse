from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import *
# from sklearn.pipeline import make_pipeline

df = pd.read_csv("../changed_values.csv")

X = df.drop('class', axis=1).values
Y = df['class'].values
print(X[:, 0])
X[:, 0] = preprocessing.scale(X[:, 0])
print(X[:, 0])
exit()
enc = preprocessing.OneHotEncoder()
enc.fit(X)
x_encoded = enc.transform(X).toarray()
X = np.concatenate((X[:, :1], x_encoded), axis=1)
pd.DataFrame(X).to_csv('X.csv', index=False)
pd.DataFrame(Y).to_csv('Y.csv', index=False)
random_state = 18
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
max_c = 0
max_accuracy = 0

for c in [100, 10, 1, 0.1, 0.01, 0.001]:
    print(f'{c=}')
    model = LogisticRegression(random_state=2, C=c).fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    if accuracy > max_accuracy:
        max_c = c
        max_accuracy = accuracy
    log_loss_ = log_loss(Y_test, Y_pred)
    print(f'{accuracy=}')
    print(f'{log_loss_=}')
    print("-" * 15)

print(f"Best accuracy {max_accuracy} with C = {max_c}")