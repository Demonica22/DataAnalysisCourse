import matplotlib.pyplot as plt
from data_generator import DataGenerator

col = 2
N = 10000
X, Y, class0, class1 = DataGenerator.onlinear_dataset_N(N)
figure, axis = plt.subplots(2, 2)

axis[0, 0].set_title(f"Гистограмма X")
axis[0, 0].set_xlabel("value")
axis[0, 0].set_ylabel("frequency")
axis[0, 0].hist(class0[:, 0], bins='auto', alpha=0.7,
                label='5 угольник')  # параметр alpha позволяет задать прозрачность цвета\
axis[0, 0].hist(class1[:, 0], bins='auto', alpha=0.7,
                label='Овал')  # параметр alpha позволяет задать прозрачность цвета

axis[1, 1].set_title(f"Гистограмма Y")
axis[1, 1].set_xlabel("value")
axis[1, 1].set_ylabel("frequency")
axis[1, 1].hist(class0[:, 1], bins='auto', alpha=0.7,
                label='5 угольник')  # параметр alpha позволяет задать прозрачность цвета
axis[1, 1].hist(class1[:, 1], bins='auto', alpha=0.7,
                label='Овал')  # параметр alpha позволяет задать прозрачность цвета
axis[1, 0].set_title(f"Скаттерограмма")
axis[1, 0].set_xlabel("x")
axis[1, 0].set_ylabel("y")
axis[1, 0].scatter(class0[:, 0], class0[:, 1], marker=".", alpha=0.7, label='5 угольник')
axis[1, 0].scatter(class1[:, 0], class1[:, 1], marker=".", alpha=0.7, label='Овал')

figure.legend(['Класс 0', 'Класс 1'])
plt.savefig("test_jpg.png")
plt.show()
