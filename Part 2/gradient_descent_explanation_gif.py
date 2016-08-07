import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np


STEP_COUNT = 25
STEP_SIZE = 0.1  # Скорость обучения
X = [i for i in np.linspace(-10, 20, 10000)]


def func(x):
    return (x - 5) ** 2 + 50 * np.sin(x) + 50

Y = [func(x) for x in X]


def func_derivative(x):
    return 2 * (x + 25 * np.cos(x) - 5)

_dy_dx = [-func_derivative(x) for x in X]


def init_gradient(num, points, line):
    print(num, points, line)
    return points, line

# Какая-то жажа и первый кадр пропускается
skip_first = True
def draw_gradient_points(num, points, line, cost_caption, step_caption, theta_caption):
    global previous_x, skip_first, ax
    if skip_first:
        skip_first = False
        return points, line
    current_x = previous_x - STEP_SIZE * func_derivative(previous_x)
    step_caption.set_text("Step: " + str(num))
    cost_caption.set_text("Func value=" + format(func(current_x), ".3f"))
    theta_caption.set_text("$\\theta$=" + format(current_x, ".3f"))
    print("Step:", num, "Previous:", previous_x, "Current", current_x)
    points[0].set_data(previous_x, func(previous_x))
    # points[1].set_data(current_x, func(current_x))
    # points.set_data([previous_x, current_x], [func(previous_x), func(current_x)])
    line.set_data([previous_x, current_x], [func(previous_x), func(current_x)])

    # if np.abs(func(previous_x) - func(current_x)) < 0.5:
    #     ax.axis([4, 6, 0, 1])
    #
    # if np.abs(func(previous_x) - func(current_x)) < 0.1:
    #     ax.axis([4.5, 5.5, 0, 0.5])
    #
    # if np.abs(func(previous_x) - func(current_x)) < 0.01:
    #     ax.axis([4.9, 5.1, 0, 0.08])

    previous_x = current_x
    return points, line


previous_x = 0
fig, ax = plt.subplots()
p = ax.get_position()
ax.set_position([p.x0 + 0.1, p.y0, p.width * 0.9, p.height])
ax.set_xlabel("$\\theta$", fontsize=18)
ax.set_ylabel("$f(\\theta)$", fontsize=18)

ax.plot(X, Y, '-r', linewidth=2.0)
# ax.plot(X, _dy_dx, '-k')
ax.axvline(5, color='black', linestyle='--')

start_point, = ax.plot([], 'bo', markersize=10.0)
end_point, = ax.plot([], 'ro')

rate_capt = ax.text(-0.3, 1.05, "Rate: " + str(STEP_SIZE), fontsize=18, transform=ax.transAxes)
step_caption = ax.text(-0.3, 1, "Step: ", fontsize=16, transform=ax.transAxes)
cost_caption = ax.text(-0.3, 0.95, "Func value: ", fontsize=12, transform=ax.transAxes)
theta_caption = ax.text(-0.3, 0.9, "$\\theta$=", fontsize=12, transform=ax.transAxes)

points = (start_point, end_point)
line, = ax.plot([], 'g--')


gradient_anim = anim.FuncAnimation(fig, draw_gradient_points, frames=STEP_COUNT,
                                   fargs=(points, line, cost_caption, step_caption, theta_caption),
                                   interval=1500)

# Для того, чтобы получить гифку необходимо установить ImageMagick
# Можно получить .mp4 файл без всяких magick-shmagick
gradient_anim.save("images/bad_func_0_1.gif", writer="imagemagick")
