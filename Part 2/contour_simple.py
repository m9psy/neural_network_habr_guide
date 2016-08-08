import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np


STEP_COUNT = 25
STEP_SIZE = 0.005  # Скорость обучения
X = np.array([i for i in np.linspace(-10, 10, 1000)])
Y = np.array([i for i in np.linspace(-10, 10, 1000)])


def func(X, Y):
    return 4 * (X ** 2) + 16 * (Y ** 2)


def dx(x):
    return 8 * x


def dy(y):
    return 32 * y

# Какая-то жажа и первый кадр пропускается
skip_first = True
def draw_gradient_points(num, point, line):
    global previous_x, previous_y, skip_first, ax
    if skip_first:
        skip_first = False
        return point
    current_x = previous_x - STEP_SIZE * dx(previous_x)
    current_y = previous_y - STEP_SIZE * dy(previous_y)
    print("Step:", num, "CurX:", current_x, "CurY", current_y, "Fun:", func(current_x, current_y))
    point.set_data([current_x], [current_y])
    # Blah-blah
    new_x = list(line.get_xdata()) + [previous_x, current_x]
    new_y = list(line.get_ydata()) + [previous_y, current_y]
    line.set_xdata(new_x)
    line.set_ydata(new_y)

    previous_x = current_x
    previous_y = current_y
    return point


previous_x, previous_y = 8.8, 8.5
fig, ax = plt.subplots()
p = ax.get_position()
ax.set_position([p.x0 + 0.1, p.y0, p.width * 0.9, p.height])
ax.set_xlabel("X", fontsize=18)
ax.set_ylabel("Y", fontsize=18)

X, Y = np.meshgrid(X, Y)
plt.contour(X, Y, func(X, Y))
point, = plt.plot([8.8], [8.5], 'bo')
line, = plt.plot([], color='black')


gradient_anim = anim.FuncAnimation(fig, draw_gradient_points, frames=STEP_COUNT,
                                   fargs=(point, line),
                                   interval=1500)

# Для того, чтобы получить гифку необходимо установить ImageMagick
# Можно получить .mp4 файл без всяких magick-shmagick
gradient_anim.save("images/contour_plot.gif", writer="imagemagick")
