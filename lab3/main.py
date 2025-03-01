import numpy as np
from pathlib import Path
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier
from lab1.data_generator import DataGenerator
import matplotlib.pyplot as plt


def calculate_metrics(prediction, answers):
    true_positive_test = np.sum((prediction == 1) & (answers == 1))
    true_negative_test = np.sum((prediction == 0) & (answers == 0))
    false_positive_test = np.sum((prediction == 1) & (answers == 0))
    false_negative_test = np.sum((prediction == 0) & (answers == 1))
    sensivity = true_positive_test / (true_positive_test + false_negative_test)
    specifity = true_negative_test / (true_negative_test + false_positive_test)
    accuracy = sum(prediction == answers) / len(answers)

    return sensivity, specifity, accuracy


with_graphs = True
x_train_path = Path("x_train.npy")
x_test_path = Path("x_test.npy")
y_train_path = Path("y_train.npy")
y_test_path = Path("y_test.npy")

path_to_files = [x_train_path,
                 x_test_path,
                 y_train_path,
                 y_test_path
                 ]
mu0 = [0, 1]
mu1 = [1, 3]
sigma0 = [1, 1]
sigma1 = [1, 2]
col = len(mu0)
N = 1000

if not all([e.exists() for e in path_to_files]):

    X, Y, *_ = DataGenerator.norm_dataset((mu0, mu1), (sigma0, sigma1), N)
    # X, Y, *_ = DataGenerator.nonlinear_dataset_N(N)

    train_size = 0.7
    trainCount = round(train_size * N * 2)  # *2 потому что было 2 класса
    Xtrain = X[0:trainCount]
    Xtest = X[trainCount:N * 2 + 1]
    Ytrain = Y[0:trainCount]
    Ytest = Y[trainCount:N * 2 + 1]
    np.save(x_train_path, Xtrain)
    np.save(x_test_path, Xtest)
    np.save(y_train_path, Ytrain)
    np.save(y_test_path, Ytest)
else:
    Xtrain = np.load(x_train_path)
    Xtest = np.load(x_test_path)
    Ytrain = np.load(y_train_path)
    Ytest = np.load(y_test_path)
max_i = 0
max_auc = 0
for i in range(10, 300, 10):
    """
    Опытным путем было получено что глубина дерева 24 - топ вариант
    """
    # if with_graphs:
    #     X = Xtrain
    #     figure, axis = plt.subplots(3)
    #     axis[0].set_title(f"Гистограмма")
    #     axis[0].hist(X[Ytrain == 0][:, 0], bins='auto', alpha=0.7)
    #     axis[0].hist(X[Ytrain == 1][:, 0], bins='auto', alpha=0.7)
    #     axis[0].set_xlabel("value")
    #     axis[0].set_ylabel("frequency")
    #     axis[1].set_title(f"Гистограмма")
    #     axis[1].hist(X[Ytrain == 0][:, 1], bins='auto', alpha=0.7)
    #     axis[1].hist(X[Ytrain == 1][:, 1], bins='auto', alpha=0.7)
    #     axis[1].set_xlabel("value")
    #     axis[1].set_ylabel("frequency")
    #     axis[2].set_xlabel("x")
    #     axis[2].set_ylabel("y")
    #     axis[2].set_title(f"Скаттерограмма")
    #     axis[2].scatter(X[Ytrain == 0][:, 0], X[Ytrain == 0][:, 1], marker=".", alpha=0.7)
    #     axis[2].scatter(X[Ytrain == 1][:, 0], X[Ytrain == 1][:, 1], marker=".", alpha=0.7)
    #
    #     figure.legend(['Класс 0', 'Класс 1'])
    #     plt.show()
    # clf = DecisionTreeClassifier(max_depth=3).fit(Xtrain, Ytrain)
    # print(clf.get_depth())
    #
    clf = RandomForestClassifier(random_state=0, n_estimators=i).fit(Xtrain, Ytrain)
    pred_test = clf.predict(Xtest)
    # print(pred_test)
    pred_test_proba = clf.predict_proba(Xtest)

    pred_train = clf.predict(Xtrain)
    pred_train_proba = clf.predict_proba(Xtrain)
    # print(pred_test_proba)

    acc_train = clf.score(Xtrain, Ytrain)
    # print(acc_train)
    acc_test = clf.score(Xtest, Ytest)
    # print(acc_test)
    # acc_test = sum(pred_test == Ytest) / len(Ytest)
    # print(acc_test)
    # from sklearn.calibration import calibration_curve
    # print(Ytest.shape)
    # y_means, proba_means = calibration_curve(Ytest, pred_test_proba, n_bins=10)

    figure, axis = plt.subplots(2)
    # print(pred_train_proba)
    # print(Ytrain)
    axis[0].hist(pred_test_proba[~Ytest, 1], bins='auto', alpha=0.7)

    axis[0].hist(pred_test_proba[Ytest, 1], bins='auto', alpha=0.7)
    axis[0].set_xlabel("Вероятность")
    axis[0].set_ylabel("Количество")
    axis[0].set_title("Результаты классификации, тест")

    # print(pred_train_proba[Ytrain, 1])
    # print(pred_train_proba[~Ytrain, 1])
    axis[1].hist(pred_train_proba[~Ytrain, 1], bins='auto', alpha=0.7)
    axis[1].hist(pred_train_proba[Ytrain, 1], bins='auto', alpha=0.7)
    axis[1].set_xlabel("Вероятность")
    axis[1].set_ylabel("Количество")
    axis[1].set_title("Результаты классификации, трейн")
    figure.legend(['Класс 0', 'Класс 1'])
    # plt.show()
    # print(Ytest.size)
    # print(calculate_metrics(pred_test, Ytest))
    # print(Ytrain.size)
    # print(calculate_metrics(pred_train, Ytrain))
    #
    # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
    # tree.plot_tree(clf.estimators_[0],
    #                filled = True)
    # fig.savefig('rf_individualtree.png')
    # fig.show()
    skplt.metrics.plot_roc_curve(Ytest, pred_test_proba, figsize=(10, 10))
    # plt.show()

    # Расчет площади под кривой
    AUC = roc_auc_score(Ytest, pred_test_proba[:, 1])
    print(i)
    print("AUC tree:" + str(AUC))
    if AUC > max_auc:
        max_auc = AUC
        max_i = i
print(max_i)
print(max_auc)
"""
600
(np.float64(0.7185430463576159), np.float64(0.7181208053691275), np.float64(0.7183333333333334))
1400
(np.float64(1.0), np.float64(1.0), np.float64(1.0))

Дерево идеально классифицирует данные, на которых было обучено, при этом на тестовых данных метрики падают до 71%


600
(np.float64(0.7384105960264901), np.float64(0.8288590604026845), np.float64(0.7833333333333333))
1400
(np.float64(1.0), np.float64(1.0), np.float64(1.0))

Для леса результаты работы на трейн выборке не изменились. А вот для тестовой выборки стали немного лучше чем у дерева.


AUC tree:0.852221209831548
"""
