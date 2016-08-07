import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np


STEP_COUNT = 10
STEP_SIZE = 0.9  # Скорость обучения
X = [i for i in np.arange(0, 10, 0.25)]


def func(x):
    return (x - 5) ** 2

Y = [func(x) for x in X]


def func_derivative(x):
    return 2 * (x - 5)


def init_gradient(num, points, line):
    print(num, points, line)
    return points, line

skip_first = True
def draw_gradient_points(num, points, line):
    global previous_x, skip_first, ax
    if skip_first:
        skip_first = False
        return points, line
    current_x = previous_x - STEP_SIZE * func_derivative(previous_x)
    print(num, previous_x, current_x)
    points.set_data([previous_x, current_x], [func(previous_x), func(current_x)])
    line.set_data([previous_x, current_x], [func(previous_x), func(current_x)])

    if np.abs(func(previous_x) - func(current_x)) < 0.5:
        ax.axis([4, 6, 0, 1])

    previous_x = current_x
    return points, line


previous_x = 0
fig, ax = plt.subplots()
ax.plot(X, Y, '-r', linewidth=2.0)
ax.axvline(5, color='black', linestyle='--')

points, = ax.plot([], 'go')
line, = ax.plot([], 'g--')


gradient_anim = anim.FuncAnimation(fig, draw_gradient_points, frames=STEP_COUNT, fargs=(points, line), interval=3000)
gradient_anim.save("azaza.mp4")

