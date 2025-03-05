import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 20)

df = pd.read_csv('online_gaming_behavior_dataset.csv')
plt.subplots(3, 4)
plt.figure(figsize=(12, 12))
df.drop("PlayerID", axis=1, inplace=True)
df.drop("Location", axis=1, inplace=True)

for i, column in enumerate(df.columns):
    plt.subplot(3, 4, i+1)
    sns.histplot(df[column], bins=10)
plt.tight_layout()
plt.show()

categorial_columns = ['Gender', 'GameGenre', 'GameDifficulty', 'EngagementLevel']
for column in categorial_columns:
    encoding = {}
    i = 0
    for genre in df[column].unique():
        encoding[genre] = i
        i += 1
    print(column)
    print(encoding)
    df[column] = df[column].replace(encoding)
df.to_csv("ready_data.csv",index=False)
