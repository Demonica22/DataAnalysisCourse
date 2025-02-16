import numpy as np
import numpy.typing as np_t
import pandas as pd

part = 3

if part == 1:
    group_number = 13
    surname_letter_number = 19
    uniform_array = np.random.rand(group_number, surname_letter_number)
    normal_array = np.random.randn(group_number, surname_letter_number)
    print(f"Массив равномерно распределенных случайных чисел:\n {uniform_array}")
    print(f"Массив нормально распределенных случайных чисел:\n {normal_array}")


    def array_info(array: np_t.NDArray) -> None:
        print(f"Shape: {array.shape}")
        print(f"Ndim: {array.ndim}")
        print(f"Data type: {array.dtype.name}")
        print(f"Item Size: {array.itemsize}")
        print(f"Size: {array.size}")


    array_info(uniform_array)
    print("-" * 15)
    array_info(normal_array)

elif part == 2:
    df = pd.read_csv('../diabetes_data_upload.csv')
    # For the most part, pandas uses NumPy arrays and dtypes for Series or individual columns of a DataFrame.\
    print(df.dtypes)
    print([np.dtype(elem).name for elem in df.dtypes])

    column_types_df = (pd.DataFrame({'Column': df.dtypes.index,
                                     'DType': df.dtypes})
                       .to_csv('column_types.csv', index=False))
elif part == 3:
    df = pd.read_csv('../diabetes_data_upload.csv')
    df.replace({"No": 0, "Negative": 0, "Yes": 1, "Positive": 1}, inplace=True)
    df.replace({"Male":0, "Female":1}, inplace=True) # Надо ли?
    df.to_csv('changed_values.csv', index=False)
    print(df)
    column_types_df = (pd.DataFrame({'Column': df.dtypes.index,
                                     'DType': df.dtypes})
                       .to_csv('column_types.csv', index=False))
