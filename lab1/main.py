import matplotlib.pyplot as plt
from data_generator import DataGenerator

mu0 = [0, 2, 3]
mu1 = [3, 5, 1]
sigma0 = [2, 1, 2]
sigma1 = [1, 2, 1]
col = len(mu0)
N = 1000
X, Y, class0, class1 = DataGenerator.norm_dataset((mu0, mu1), (sigma0, sigma1), N)

train_size = 0.7
trainCount = round(train_size * N * 2)  # *2 потому что было 2 класса
Xtrain = X[0:trainCount]
Xtest = X[trainCount:N * 2 + 1]
Ytrain = Y[0:trainCount]
Ytest = Y[trainCount:N * 2 + 1]

if __name__ == "__main__":
    figure, axis = plt.subplots(col, 2)
    for i in range(0, col):
        axis[i, 0].set_title(f"Гистограмма {i + 1}")
        hist_class0 = axis[i, 0].hist(class0[:, i], bins='auto', alpha=0.7,
                                      label='Класс 0')
        hist_class1 = axis[i, 0].hist(class1[:, i], bins='auto', alpha=0.7, label='Класс 1')
        axis[i, 0].set_xlabel("value")
        axis[i, 0].set_ylabel("frequency")
        axis[i, 1].set_xlabel("x")
        axis[i, 1].set_ylabel("y")
        if i != col - 1:
            axis[i, 1].set_title(f"Скаттерограмма {i + 1} и {i + 2}")
            axis[i, 1].scatter(class0[:, i], class0[:, i + 1], marker=".", alpha=0.7, label='Класс 0')
            axis[i, 1].scatter(class1[:, i], class1[:, i + 1], marker=".", alpha=0.7, label='Класс 1')
        else:
            axis[i, 1].set_title(f"Скаттерограмма {i + 1} и {1}")

            axis[i, 1].scatter(class0[:, i], class0[:, 0], marker=".", alpha=0.7, label='Класс 0')
            axis[i, 1].scatter(class1[:, i], class1[:, 0], marker=".", alpha=0.7, label='Класс 1')

    figure.legend(['Класс 0', 'Класс 1'])
    plt.show()
