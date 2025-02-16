import numpy as np
from typing import Any
from matplotlib.path import Path


def is_point_in_pentagon(x, y, rect_x, rect_y, rect_width, rect_height):
    # Определяем вершины пятиугольника
    bottom_left = (rect_x, rect_y)
    bottom_right = (rect_x + rect_width, rect_y)
    top_right = (rect_x + rect_width, rect_y + rect_height)
    top_left = (rect_x, rect_y + rect_height)
    center_right = (rect_x + rect_width / 2, rect_y + rect_height / 2)

    pentagon_vertices = [bottom_left, bottom_right, center_right, top_right, top_left, bottom_left]

    pentagon_path = Path(np.array(pentagon_vertices))

    return pentagon_path.contains_point((x, y))  # , pentagon_vertices


def is_point_in_ellipse(x, y, center_x, center_y, width, height):
    norm_x = (x - center_x) / (width / 2)
    norm_y = (y - center_y) / (height / 2)
    return norm_x ** 2 + norm_y ** 2 <= 1


class DataGenerator:
    @staticmethod
    def norm_dataset(mu: tuple[list[int], list[int]],
                     sigma: tuple[list[int], list[int]],
                     N: int
                     ) -> \
            tuple[np.ndarray[np.floating[np.float64], Any],
            np.ndarray[np.floating[np.float64], Any],
            np.ndarray[np.floating[np.float64], Any],
            np.ndarray[np.floating[np.float64], Any]]:
        mu0 = mu[0]
        mu1 = mu[1]
        sigma0 = sigma[0]
        sigma1 = sigma[1]

        col = len(mu0)
        class0 = np.random.normal(mu0[0], sigma0[0], [N, 1])  # инициализируем первый столбец

        class1 = np.random.normal(mu1[0], sigma1[0], [N, 1])
        for i in range(1, col):
            v0 = np.random.normal(mu0[i], sigma0[i], [N, 1])
            class0 = np.hstack((class0, v0))
            v1 = np.random.normal(mu1[i], sigma1[i], [N, 1])
            class1 = np.hstack((class1, v1))
        X = np.vstack((class0, class1))
        # print(X)
        # print(X.size)
        Y0 = np.zeros((N, 1), dtype=bool)
        Y1 = np.ones((N, 1), dtype=bool)
        Y = np.vstack((Y0, Y1)).ravel()

        rng = np.random.default_rng()
        arr = np.arange(2 * N)  # индексы для перемешивания [ 0, 1 ,2 .... 1999]
        rng.shuffle(arr)
        X = X[arr]
        Y = Y[arr]

        return X, Y, class0, class1

    @staticmethod
    def onlinear_dataset_N(num_points):
        """
        Генерирует два массива точек, распределенных в пределах пятиугольника и эллипса.
        """
        pentagon_params = (0, 0, 10, 6)  # x, y, width, height
        ellipse_params = (17, 3, 10, 4)  # center_x, center_y, width, height

        pentagon_points = []
        ellipse_points = []

        x_min, x_max = min(pentagon_params[0], ellipse_params[0] - ellipse_params[2] / 2), max(
            pentagon_params[0] + pentagon_params[2], ellipse_params[0] + ellipse_params[2] / 2)
        y_min, y_max = min(pentagon_params[1], ellipse_params[1] - ellipse_params[3] / 2), max(
            pentagon_params[1] + pentagon_params[3], ellipse_params[1] + ellipse_params[3] / 2)

        while len(pentagon_points) < num_points or len(ellipse_points) < num_points:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)

            if len(pentagon_points) < num_points and is_point_in_pentagon(x, y, *pentagon_params):
                pentagon_points.append((x, y))

            if len(ellipse_points) < num_points and is_point_in_ellipse(x, y, *ellipse_params):
                ellipse_points.append((x, y))
        class0 = np.array(pentagon_points)
        class1 = np.array(ellipse_points)
        X = np.vstack((class0,class1))
        Y0 = np.zeros((num_points, 1), dtype=bool)
        Y1 = np.ones((num_points, 1), dtype=bool)
        Y = np.vstack((Y0, Y1)).ravel()

        rng = np.random.default_rng()
        arr = np.arange(num_points)
        rng.shuffle(arr)
        X = X[arr]
        Y = Y[arr]
        return X, Y, class0, class1
