import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import keras
import time

for lib in [np, pd, sklearn, tf, keras]:
    print(lib.__name__, lib.__version__)
print("-" * 15)

start = time.time()

df = pd.read_csv('../diabetes_data_upload.csv')
print(f"Avarage age is: {df['Age'].mean()}")
avarages = df.groupby('Gender')['Age'].mean()

print(f"Avarage age for female is: {avarages['Female']}")
print(f"Avarage age for male is: {avarages['Male']}")
print(f"Execution time: {time.time() - start}")

'''
pyre python
Avarage age is: 48.02884615384615
Avarage age for female is: 47.03125
Avarage age for male is: 48.61280487804878
Execution time: 0.0010006427764892578

pandas:
Avarage age is: 48.02884615384615
Avarage age for female is: 47.03125
Avarage age for male is: 48.61280487804878
Execution time: 0.0019998550415039062

'''
