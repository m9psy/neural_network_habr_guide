import numpy as np
import matplotlib.pyplot as plt

TOTAL = 10
DIMENSIONS = 2
STEP = 2


def generate_random_plane():
    return np.random.uniform(low=-DIMENSIONS, high=DIMENSIONS, size=DIMENSIONS)
HYPER_PLANE = np.array([2, 5])


def func(X):
    return X @ HYPER_PLANE


def linear_hypothesis(X, theta):
    return X @ theta


def generate_sample(X):
    return func(X) + np.random.normal()


def adagrad():
    pass


def OLS_cost(X, Y, theta, hypothesis=linear_hypothesis):
    hyp = hypothesis(X=X, theta=theta)
    return (Y - hyp).T @ (Y - hyp)


def OLS_derivative(X, Y, theta, hypothesis=linear_hypothesis):
    h = hypothesis(X, theta)
    return -2 * (X.T @ (Y - h))


def SGD_minimizer(theta, derivative, step):
    theta_dif = step * derivative
    return theta - theta_dif


def solve_theta(minimizer=SGD_minimizer, cost_function=OLS_cost,
                cost_function_derivative=OLS_derivative, gradient_step=0.05):
    # Начальное значени любое
    theta = np.zeros((DIMENSIONS, 1))
    # Предел сходимости, easy mode
    epsilon = 10 ** -5
    memory = 0
    current = cost_function(X, Y, theta)
    cost_function_values = [current[0][0]]
    # Повторять, пока метод не сойдется до указанной точности
    while np.abs(current - memory) >= epsilon:
        memory = current
        print("COST: ", current)
        current_derivative = cost_function_derivative(X=X, Y=Y, theta=theta)
        # print("Deriv: ", current_derivative)
        theta = minimizer(theta=theta, derivative=current_derivative, step=gradient_step)
        # print("Theta: ", theta)
        current = cost_function(X, Y, theta)
        cost_function_values.append(current[0][0])
        # plt.plot(X[:, 1:], Y, 'bo')
        # plt.plot(X[:, 1:], linear_hypothesis(X, theta), 'g', linewidth=2.0)
        # plt.show()
    print("Finished with ", theta)
    print("Real params: ", np.linalg.pinv(X).dot(Y))
    plt.plot(cost_function_values)
    plt.show()


X = np.zeros((TOTAL, DIMENSIONS), dtype=np.float32)
# x0 = 1, нету его
X[:, 0] = 1
Y = np.zeros((TOTAL, 1), dtype=np.float32)
for i, x in enumerate(X):
    Y[i] = generate_sample(x)
    if i != TOTAL - 1:
        X[i + 1, 1:] = X[i, 1:] + STEP

X[:, 1] = (X[:, 1] - np.min(X[:, 1])) / (np.max(X[:, 1]) - np.min(X[:, 1]))

# T1 = np.linspace(0, 50, TOTAL)
# T2 = np.linspace(0, 100, TOTAL)
# COST = []
#
# for i in range(len(T1)):
#     cost = OLS_cost(X, Y, np.array([[T1[i]], [T2[i]]]))
#     COST.append(cost[0][0])
#
# print("fished")
#
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# Xgrid, Ygrid = np.meshgrid(T1, T2)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# print("drawing")
# ax.plot_surface(Xgrid, Ygrid, COST, rstride=1, cstride=1, cmap=cm.coolwarm)
# plt.show()

solve_theta()
