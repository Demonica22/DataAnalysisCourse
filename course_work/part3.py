import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('ready_data.csv')
df.drop(index=0, inplace=True, axis=1)
columns = ['Gender', 'GameDifficulty', 'EngagementLevel', 'GameGenre']
unique_values = {feature: df[feature].unique() for feature in columns}
correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot=True, fmt=".2f", linewidths=0.5)
plt.title("Корреляционная матрица")
plt.show()
