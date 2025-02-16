import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def is_point_in_ellipse(x, y, center_x, center_y, width, height):
    norm_x = (x - center_x) / (width / 2)
    norm_y = (y - center_y) / (height / 2)
    return norm_x ** 2 + norm_y ** 2 <= 1


# Параметры эллипса
center_x, center_y = 5, 5  # Центр эллипса
width, height = 10, 6  # Размеры эллипса

test_points = [(5, 5), (2, 6), (8, 5), (10, 7), (5, 8)]

# Проверка точек и отображение
fig, ax = plt.subplots()
ellipse = Ellipse((center_x, center_y), width=width, height=height, color='b', fill=False, label="Ellipse")
ax.add_patch(ellipse)

inside_points = []
outside_points = []

for tx, ty in test_points:
    if is_point_in_ellipse(tx, ty, center_x, center_y, width, height):
        inside_points.append((tx, ty))
    else:
        outside_points.append((tx, ty))

# Отображение точек
if inside_points:
    inside_points = np.array(inside_points)
    ax.scatter(inside_points[:, 0], inside_points[:, 1], c='g', label="Inside Points")
if outside_points:
    outside_points = np.array(outside_points)
    ax.scatter(outside_points[:, 0], outside_points[:, 1], c='r', label="Outside Points")

ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.set_aspect('equal')
ax.legend()
plt.grid()
plt.show()
