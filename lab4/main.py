import numpy as np
from matplotlib import pyplot as plt
from lab1.data_generator import DataGenerator


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_derivative(p):
    return p * (1 - p)


class NeuralNetwork:

    def __init__(self, x, y):
        self.input = x
        n_inp = self.input.shape[1]  # кол-во входов
        n_neuro = 4  # число нейронов на главном слое
        # инициализация весов рандомными значениями
        self.weights1 = np.random.rand(n_inp, n_neuro)
        self.weights2 = np.random.rand(n_neuro, 1)

        self.weights_history = []

        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        # выходы слоёв вычисляются по сигмоиде
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backprop(self):
        # здесь происходит коррекция весов по известному вам из курса алгоритму. Подумайте, что значит .T после NN.weights2
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                 self.weights2.T) * sigmoid_derivative(self.layer1))
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))

        # обновляем веса
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights_history.append((self.weights1, self.weights2))
    def train(self, X, y):
        # весь процесс обучения прост – высчитываем выход с помощью прямого распространения, а после обновляем веса
        self.output = self.feedforward()
        self.backprop()


mu0 = [0, 2, 3]  # параметры выборки даны для примера, задайте свои и можно выбрать более 3х столбцов
mu1 = [3, 5, 1]
sigma0 = [2, 1, 2]
sigma1 = [1, 2, 1]
N = 1000  # число объектов класса
col = len(mu0)  # количество столбцов-признаков

mu = (mu0, mu1)
sigma = (sigma0, sigma1)
X, Y, class0, class1 = DataGenerator.norm_dataset(mu, sigma, N)
Y = np.reshape(Y, [2000, 1])
NN = NeuralNetwork(X, Y)  # инициализируем сетку на наших данных
N_epoch = 100
losses = []

for i in range(N_epoch):
    print("for iteration # " + str(i))
    loss = np.mean(np.square(Y - NN.feedforward()))
    losses.append(loss)
    print("Loss:" + str(loss))  # потери рассчитайте как среднеквадратичные
    NN.train(X, Y)
print(NN.weights_history)
pred = NN.feedforward()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
print(losses)
print(range(1, N_epoch + 1))
plt.plot(range(1, N_epoch + 1), losses)
plt.show()
