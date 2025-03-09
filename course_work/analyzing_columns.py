import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('online_gaming_behavior_dataset.csv')
df.drop('PlayerID', axis=1, inplace=True)
columns_to_analyze = ['Age', 'PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes', 'PlayerLevel',
                      'AchievementsUnlocked']
plt.subplots(1, 1)
for i, column in enumerate(columns_to_analyze):
    plt.subplot(1, 1, 1)
    sns.boxplot(df[column])
plt.tight_layout()


plt.show()
plt.cla()

# columns_to_analyze = ['Gender', 'GameGenre', 'InGamePurchases']
#
# plt.subplots(1, len(columns_to_analyze))
# for i, column in enumerate(columns_to_analyze):
#     plt.subplot(1, len(columns_to_analyze), i + 1)
#     df[column].value_counts().plot.pie(autopct='%1.1f%%',
#                                        startangle=90,
#                                        explode=[0.05] * df[column].nunique()
#                                        )
#     plt.title(f'{column}')
#     plt.ylabel('')
# plt.tight_layout()
# plt.show()
#
# numerical_columns = df.select_dtypes(include=['int64', 'float64'])
# correlation_matrix = numerical_columns.corr()
#
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", linewidths=0.5)
# plt.title("Корреляционная матрица")
# plt.show()
