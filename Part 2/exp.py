import numpy as np
import matplotlib.pyplot as plt

TOTAL = 200
STEP = 0.125


def func(x):
    return x + 0.5 * (x ** 2) + 200 * np.sin(x)


def generate_sample(total=TOTAL):
    x = 0
    while x < total * STEP:
        yield func(x) + np.random.normal(loc=0, scale=60)
        x += STEP


X = np.arange(0, TOTAL * STEP, STEP)
Y = np.array([y for y in generate_sample(TOTAL)])
Y_real = np.array([func(x) for x in X])


plt.plot(X, Y, 'bo')
plt.plot(X, Y_real, '-r', linewidth=2.0)
plt.show()
