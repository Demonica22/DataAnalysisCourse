import matplotlib.path as mplPath
import numpy as np
import matplotlib.pyplot as plt


def is_point_in_pentagon(x, y, rect_x, rect_y, rect_width, rect_height):

    bottom_left = (rect_x, rect_y)
    bottom_right = (rect_x + rect_width, rect_y)
    top_right = (rect_x + rect_width, rect_y + rect_height)
    top_left = (rect_x, rect_y + rect_height)
    center_right = (rect_x + rect_width / 2, rect_y + rect_height / 2)

    pentagon_vertices = [bottom_left, bottom_right, center_right, top_right, top_left, bottom_left]


    pentagon_path = mplPath.Path(np.array(pentagon_vertices))

    return pentagon_path.contains_point((x, y)), pentagon_vertices



rect_x, rect_y = 0, 0  # Координаты нижнего левого угла прямоугольника
rect_width, rect_height = 10, 6  # Размеры прямоугольника

test_points = [(5, 3), (2, 2), (8, 5), (11, 3), (5, 7)]


fig, ax = plt.subplots()
inside_points = []
outside_points = []

for tx, ty in test_points:
    inside, pentagon_vertices = is_point_in_pentagon(tx, ty, rect_x, rect_y, rect_width, rect_height)
    if inside:
        inside_points.append((tx, ty))
    else:
        outside_points.append((tx, ty))


pentagon_vertices = np.array(pentagon_vertices)
ax.plot(pentagon_vertices[:, 0], pentagon_vertices[:, 1], 'b-', label="Pentagon")


if inside_points:
    inside_points = np.array(inside_points)
    ax.scatter(inside_points[:, 0], inside_points[:, 1], c='g', label="Inside Points")
if outside_points:
    outside_points = np.array(outside_points)
    ax.scatter(outside_points[:, 0], outside_points[:, 1], c='r', label="Outside Points")

ax.legend()
ax.set_xlim(-2, 12)
ax.set_ylim(-2, 10)
plt.grid()
plt.show()
