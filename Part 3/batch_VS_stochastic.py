import numpy as np
import matplotlib.pyplot as plt

TOTAL = 200
STEP = 0.25
EPS = 0.1
INITIAL_THETA = [9, 14]


def func(x):
    return 0.2 * x + 3


def generate_sample(total=TOTAL):
    x = 0
    while x < total * STEP:
        yield func(x) + np.random.uniform(-1, 1) * np.random.uniform(2, 8)
        x += STEP


def cost_function(A, Y, theta):
    return (Y - A@theta).T@(Y - A@theta)


def batch_descent(A, Y, speed=0.001):
    theta = np.array(INITIAL_THETA.copy(), dtype=np.float32)
    theta.reshape((len(theta), 1))
    previous_cost = 10 ** 6
    current_cost = cost_function(A, Y, theta)
    while np.abs(previous_cost - current_cost) > EPS:
        previous_cost = current_cost
        derivatives = [0] * len(theta)
        # ---------------------------------------------
        for j in range(len(theta)):
            summ = 0
            for i in range(len(Y)):
                summ += (Y[i] - A[i]@theta) * A[i][j]
            derivatives[j] = summ
        # Выполнение требования одновремменности
        theta[0] += speed * derivatives[0]
        theta[1] += speed * derivatives[1]
        # ---------------------------------------------
        current_cost = cost_function(A, Y, theta)
        print("Batch cost:", current_cost)
        plt.plot(theta[0], theta[1], 'ro')
    return theta


def stochastic_descent(A, Y, speed=0.1):
    theta = np.array(INITIAL_THETA.copy(), dtype=np.float32)
    previous_cost = 10 ** 6
    current_cost = cost_function(A, Y, theta)
    while np.abs(previous_cost - current_cost) > EPS:
        previous_cost = current_cost
        # --------------------------------------
        # for i in range(len(Y)):
        i = np.random.randint(0, len(Y))
        derivatives = [0] * len(theta)
        for j in range(len(theta)):
            derivatives[j] = (Y[i] - A[i]@theta) * A[i][j]
        theta[0] += speed * derivatives[0]
        theta[1] += speed * derivatives[1]
        # --------------------------------------
        current_cost = cost_function(A, Y, theta)
        print("Stochastic cost:", current_cost)
        plt.plot(theta[0], theta[1], 'ro')
    return theta

X = np.arange(0, TOTAL * STEP, STEP)
Y = np.array([y for y in generate_sample(TOTAL)])

# Нормализацию вкрячил, чтобы парабалоид красивый был
X = (X - X.min()) / (X.max() - X.min())

A = np.empty((TOTAL, 2))
A[:, 0] = 1
A[:, 1] = X

theta = np.linalg.pinv(A).dot(Y)
print(theta, cost_function(A, Y, theta))

import time
start = time.clock()
theta_stochastic = stochastic_descent(A, Y, 0.1)
print("St:", time.clock() - start, theta_stochastic)

start = time.clock()
theta_batch = batch_descent(A, Y, 0.001)
print("Btch:", time.clock() - start, theta_batch)
