from mpl_toolkits.mplot3d import Axes3D
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
    # points[1].set_data(current_x, func(current_x))
    # points.set_data([previous_x, current_x], [func(previous_x), func(current_x)])
    # Велосипеды-велосипедики
    new_x = list(line.get_xdata()) + [previous_x, current_x]
    new_y = list(line.get_ydata()) + [previous_y, current_y]
    line.set_xdata(new_x)
    line.set_ydata(new_y)
    # line.set_data([previous_x, current_x], [previous_y, current_y])

    # if np.abs(func(previous_x) - func(current_x)) < 0.5:
    #     ax.axis([4, 6, 0, 1])
    #
    # if np.abs(func(previous_x) - func(current_x)) < 0.1:
    #     ax.axis([4.5, 5.5, 0, 0.5])
    #
    # if np.abs(func(previous_x) - func(current_x)) < 0.01:
    #     ax.axis([4.9, 5.1, 0, 0.08])

    previous_x = current_x
    previous_y = current_y
    return point


previous_x, previous_y = 8.8, 8.5
fig, ax = plt.subplots()
p = ax.get_position()
ax.set_position([p.x0 + 0.1, p.y0, p.width * 0.9, p.height])
ax.set_xlabel("X", fontsize=18)
ax.set_ylabel("Y", fontsize=18)

# ax.plot(X, Y, '-r', linewidth=2.0)
# ax.plot(X, _dy_dx, '-k')
# ax.axvline(5, color='black', linestyle='--')

# start_point, = ax.plot([], 'bo')
# end_point, = ax.plot([], 'ro')

# rate_capt = ax.text(-0.3, 1.05, "Rate: " + str(STEP_SIZE), fontsize=18, transform=ax.transAxes)
# step_caption = ax.text(-0.3, 1, "Step: ", fontsize=16, transform=ax.transAxes)
# cost_caption = ax.text(-0.3, 0.95, "Func value: ", fontsize=12, transform=ax.transAxes)
# theta_caption = ax.text(-0.3, 0.9, "$\\theta$=", fontsize=12, transform=ax.transAxes)


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
