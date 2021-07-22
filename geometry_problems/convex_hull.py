import timeit

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng


def vector_angle(a, b):
    """Returns tuple of angle in radians and vector length"""
    relative_point = b[1] - a[1], b[0] - a[0]  # Flipped x,y for arctan2
    angle = np.arctan2(*relative_point)
    return (angle, np.sqrt(relative_point[1] ** 2 + relative_point[0] ** 2))


def is_convex(a: tuple, b: tuple, c: tuple):
    """Returns True if points make a CCW turn"""
    ab = b[0] - a[0], b[1] - a[1]
    bc = c[0] - b[0], c[1] - b[1]
    return np.cross(ab, bc) > 0


def graham_scan(points: list):
    """Calculates convex hull using Graham Scan algorithm"""
    # Start point is bottom-most point, priority to lower x
    start_point = min(points, key=lambda p: (p[1], p[0]))
    # Sort points according to ccw angle from start_point
    points.sort(key=lambda p: vector_angle(start_point, p))

    if len(points) <= 3:
        return points

    stack = []
    for point in points:
        while len(stack) >= 2 and not is_convex(stack[-2], stack[-1], point):
            stack.pop()
        stack.append(point)

    if not is_convex(stack[-2], stack[-1], stack[0]):
        stack.pop()

    return stack


def monotone_chain(points: list):
    """Calculates convex hull using Monotone Chain algorithm (Andrew's algorithm)"""
    points.sort()

    if len(points) <= 3:
        return points

    upper_hull = []
    lower_hull = []

    for point in points:
        while len(lower_hull) >= 2 and not is_convex(lower_hull[-2], lower_hull[-1], point):
            lower_hull.pop()
        lower_hull.append(point)

    for point in points[::-1]:
        while len(upper_hull) >= 2 and not is_convex(upper_hull[-2], upper_hull[-1], point):
            upper_hull.pop()
        upper_hull.append(point)

    return lower_hull[:-1] + upper_hull[:-1]


def plot_hull(points: list, hull: list):
    """Scatter plot of points with hull lines drawn over"""
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        plt.plot([x1, x2], [y1, y2], color="r")
    plt.plot(*zip(*points), "ko")
    plt.show()


def monotone_chain_test(points_list):
    for points in points_list:
        monotone_chain(points)


def graham_scan_test(points_list):
    for points in points_list:
        graham_scan(points)


if __name__ == "__main__":
    point_range = 100
    num_points = 10
    num_runs = 1000

    rng = default_rng()

    points_list = []
    for _ in range(num_runs):
        xs = rng.integers(0, point_range, num_points)
        ys = rng.integers(0, point_range, num_points)
        points = [(x, y) for x, y in zip(xs, ys)]
        points_list.append(points)

    # Min time (per hull) of 10 repetitions of num_runs different hulls
    mchain_time = (
        min(
            timeit.repeat(
                "monotone_chain_test(points_list)", globals=globals(), repeat=10, number=1
            )
        )
        / num_runs
    )
    print(f"{mchain_time = }")

    gscan_time = (
        min(timeit.repeat("graham_scan_test(points_list)", globals=globals(), repeat=10, number=1))
        / num_runs
    )
    print(f"{gscan_time = }")
    print(f"mchain {round(((mchain_time / gscan_time) - 1) * 100, 2)}% slower")
