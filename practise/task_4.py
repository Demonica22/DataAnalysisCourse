import pandas as pd
import matplotlib.pyplot as plt


def info_printer(df: pd.DataFrame, in_group_number) -> None:
    print(df.info())
    print(df.describe())
    print(df.iloc[:, :in_group_number + 2].head())
    pass


if __name__ == '__main__':
    in_group_number = 8
    df = pd.read_csv('../diabetes_data_upload.csv')
    '''
    Напишите функцию, которая принимает на вход DataFrame pandas, и
    выводит в консоль информацию об индексах, типах данных, описательную
    статистику и первые 5 строк для первых N+2 столбцов матрицы (здесь и далее
    N – ваш номер в списке группы). Используйте данную функцию для описания
    дата фрейма, загруженного в пункте 1.
    
    
    '''
    # info_printer(df, in_group_number)
    '''
    Создайте два дата фрейма, полученных путем определения в один из них
    только строк со значением Yes в колонке N+1, а в другой – только со
    значением No.
    '''
    # yes_df = df[df.iloc[:, in_group_number + 1] == "Yes"]
    # print(yes_df)
    # no_df = df[df.iloc[:, in_group_number + 1] == "No"]
    # print(no_df)
    '''
    Выполните сортировку вашей исходной таблицы по нескольким ключам:
    первый ключ – колонка N+1, второй класс – колонка – N+2, третий ключ –
    возраст (Age). Сохраните результат в виде отдельного DataFrame.
    '''
    # sorted_df = df.sort_values(by=[df.columns[in_group_number + 1], df.columns[in_group_number + 2], 'Age'])
    # print(sorted_df)

    '''
    Проверьте, есть ли в ваших данных пропущенные значения и удалите
    строки, в которых есть хотя бы один пропуск.
    '''
    # if pd.isna(df).any().any():
    #     print("Удаляю NaN")
    #     df_cleaned = df.dropna(how='any', axis='index')
    #     # вроде axis и так по дефолту - строка, но в методичке написано что столбцы дропнутся
    #     # поэтому явно укажем что дроп будет по строчно - index
    #     print(df_cleaned)
    # else:
    #     print("Dataframe чист")

    # figure, axis = plt.subplots(1, 3)
    # df1 = df.iloc[:, 0:3]
    # axis[0].set_title(df.columns[0])
    # axis[0].hist(df1.iloc[:, 0])
    # axis[1].set_title(df.columns[1])
    # axis[1].hist(df1.iloc[:, 1])
    # axis[2].set_title(df.columns[2])
    # axis[2].hist(df1.iloc[:, 2])
    # plt.show()
    '''
    Постройте гистограммы распределения Age для двух таблиц,
    получившихся при разделении данных в пункте 3 (по возможности на
    соседних полях одного окна, используя функцию subplot). Проанализируйте
    данные графики и сделайте выводы.
    
    '''
    # figure, axis = plt.subplots(1, 2)
    # axis[0].set_title(f'Age for positive {df.columns[in_group_number + 1]}')
    # axis[0].hist(yes_df['Age'])
    # axis[1].set_title(f'Age for negative {df.columns[in_group_number + 1]}')
    # axis[1].hist(no_df['Age'])
    # plt.show()
    '''
    
    Постройте boxplot распределения Age для двух таблиц, получившихся
    при разделении данных в пункте 3 (оба ящика с усами должны быть на одном
    полотне, друг рядом с другом). Проанализируйте данные графики и сделайте
    выводы.
    '''
    # figure, axis = plt.subplots(1, 2)
    # axis[0].set_title(f'Age for positive {df.columns[in_group_number + 1]}')
    # axis[0].boxplot(yes_df['Age'])
    # axis[1].set_title(f'Age for negative {df.columns[in_group_number + 1]}')
    # axis[1].boxplot(no_df['Age'])
    # plt.show()

    '''
    Постройте scatter matrix для колонок Age, колонки N+1, и колонки N+2,
    закодировов цветом переменную class (Positive – красным, Negative – синим
    цветом).
    '''
    df = pd.read_csv("../changed_values.csv")
    columns = ['Age', df.columns[in_group_number + 1], df.columns[in_group_number + 2], 'class']
    columnsColor = {1: 'red', 0: 'blue'}
    colors = df['class'].map(columnsColor)
    pd.plotting.scatter_matrix(df[columns], color=colors)
    plt.show()
