import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from matplotlib.patches import Ellipse


def is_point_in_pentagon(x, y, rect_x, rect_y, rect_width, rect_height):
    """
    Проверяет, находится ли точка (x, y) внутри пятиугольника.
    """
    global pentagon_vertices

    bottom_left = (rect_x, rect_y)
    bottom_right = (rect_x + rect_width, rect_y)
    top_right = (rect_x + rect_width, rect_y + rect_height)
    top_left = (rect_x, rect_y + rect_height)
    center_right = (rect_x + rect_width / 2, rect_y + rect_height / 2)

    pentagon_vertices = [bottom_left, bottom_right, center_right, top_right, top_left, bottom_left]
    print(pentagon_vertices)
    pentagon_path = mplPath.Path(np.array(pentagon_vertices))

    return pentagon_path.contains_point((x, y))


def is_point_in_ellipse(x, y, center_x, center_y, width, height):
    """
    Проверяет, находится ли точка (x, y) внутри овала (эллипса).
    """
    norm_x = (x - center_x) / (width / 2)
    norm_y = (y - center_y) / (height / 2)
    return norm_x ** 2 + norm_y ** 2 <= 1


def onlinear_dataset_N(num_points, pentagon_params, ellipse_params):
    """
    Генерирует два массива точек, распределенных в пределах пятиугольника и эллипса.
    """
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

    return np.array(pentagon_points), np.array(ellipse_points)


# Параметры фигур
pentagon_params = (0, 0, 10, 6)  # x, y, width, height
ellipse_params = (17, 3, 10, 6)  # center_x, center_y, width, height
num_points = 200

# Генерация точек
dataset_pentagon, dataset_ellipse = onlinear_dataset_N(num_points, pentagon_params, ellipse_params)

# Визуализация данных
fig, ax = plt.subplots()
ax.scatter(dataset_pentagon[:, 0], dataset_pentagon[:, 1], c='g', label="Pentagon Points", alpha=0.6)
ax.scatter(dataset_ellipse[:, 0], dataset_ellipse[:, 1], c='r', label="Ellipse Points", alpha=0.6)
ellipse = Ellipse((ellipse_params[0], ellipse_params[1]), width=ellipse_params[2], height=ellipse_params[3], color='b',
                  fill=False, label="Ellipse")
ax.add_patch(ellipse)
pentagon_vertices = np.array(pentagon_vertices)
ax.plot(pentagon_vertices[:, 0], pentagon_vertices[:, 1], 'b-', label="Pentagon")


ax.legend()

# ax.set_xlim(-2, 12)
# ax.set_ylim(-2, 12)
ax.set_aspect('equal')
plt.grid()
plt.show()
