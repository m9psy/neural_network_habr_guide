import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

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


X = np.arange(0, TOTAL * STEP, STEP)
Y = np.array([y for y in generate_sample(TOTAL)])

# Нормализацию вкрячил, чтобы парабалоид красивый был
X = (X - X.min()) / (X.max() - X.min())

A = np.empty((TOTAL, 2))
A[:, 0] = 1
A[:, 1] = X

theta_real = np.linalg.pinv(A).dot(Y)
print(theta_real, cost_function(A, Y, theta_real))


theta = np.array(INITIAL_THETA.copy(), dtype=np.float32)
theta.reshape((len(theta), 1))
previous_cost = 10 ** 6
current_cost = cost_function(A, Y, theta)
def batch_descent(num, A, Y, point, line, speed=0.001):
    global theta, previous_cost, current_cost, ax
    previous_cost = current_cost
    derivatives = [0] * len(theta)
    # ---------------------------------------------
    for j in range(len(theta)):
        summ = 0
        for i in range(len(Y)):
            summ += (Y[i] - A[i]@theta) * A[i][j]
        derivatives[j] = summ

    # ---------------------------------------------
    print("Batch cost:", current_cost, num)
    point.set_data([theta[0]], [theta[1]])
    # --------------------------------------------------
    new_x = list(line.get_xdata()) + [theta[0], theta[0] + speed * derivatives[0]]
    new_y = list(line.get_ydata()) + [theta[1], theta[1] + speed * derivatives[1]]
    line.set_xdata(new_x)
    line.set_ydata(new_y)
    # --------------------------------------
    theta[0] += speed * derivatives[0]
    theta[1] += speed * derivatives[1]
    current_cost = cost_function(A, Y, theta)

    if num == 15:
        plt.axis([2.5, 3.5, 9, 11])

    return point

theta = np.array(INITIAL_THETA.copy(), dtype=np.float32)
theta.reshape((len(theta), 1))
previous_cost = 10 ** 6
current_cost = cost_function(A, Y, theta)
def stochastic_descent(num, A, Y, point, line, speed=0.1):
    global theta, previous_cost, current_cost, ax
    previous_cost = current_cost
    # --------------------------------------
    # for i in range(len(Y)):
    i = np.random.randint(0, len(Y))
    derivatives = [0] * len(theta)
    for j in range(len(theta)):
        derivatives[j] = (Y[i] - A[i]@theta) * A[i][j]
    current_cost = cost_function(A, Y, theta)
    print("Stochastic cost:", current_cost)

    point.set_data([theta[0]], [theta[1]])
    # --------------------------------------------------
    new_x = list(line.get_xdata()) + [theta[0], theta[0] + speed * derivatives[0]]
    new_y = list(line.get_ydata()) + [theta[1], theta[1] + speed * derivatives[1]]
    line.set_xdata(new_x)
    line.set_ydata(new_y)
    # --------------------------------------
    theta[0] += speed * derivatives[0]
    theta[1] += speed * derivatives[1]
    current_cost = cost_function(A, Y, theta)

    if num == 15:
        plt.axis([2.5, 3.5, 9, 11])

    return point


t1 = np.arange(0, 10, 0.01)
t2 = np.arange(5, 15, 0.01)

print("Generating data for contour")
t1_mesh, t2_mesh = np.meshgrid(t1, t2)
cost_array = []
counter = 0
for i in range(len(t1_mesh)):
    cost_array.append([])
    for j in range(len(t1_mesh[i])):
        cost_array[-1].append(cost_function(A, Y, np.array([t1_mesh[i][j], t2_mesh[i][j]])))

print("Fin")

cost_array = np.array(cost_array)


# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(t1_mesh, t2_mesh, cost_array, rstride=50, cstride=50, cmap=plt.get_cmap("jet"))


fig, ax = plt.subplots()
ax.set_xlabel("$\\theta_1$", fontsize=20)
ax.set_ylabel("$\\theta_2$", fontsize=20)
plt.contour(t1_mesh, t2_mesh, cost_array)
plt.plot(theta_real[0], theta_real[1], 'kx')

point, = plt.plot([INITIAL_THETA[0]], [INITIAL_THETA[1]], 'bo')
line, = plt.plot([], color='black')

gradient_anim = anim.FuncAnimation(fig, stochastic_descent, fargs=(A, Y, point, line), frames=50, interval=500)

gradient_anim.save("images/stochastic.gif", writer="imagemagick")
