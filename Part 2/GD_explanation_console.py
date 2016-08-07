STEP_COUNT = 25
STEP_SIZE = 1.2  # Скорость обучения


def func(x):
    return (x - 5) ** 2


def func_derivative(x):
    return 2 * (x - 5)

previous_x, current_x = 0, 0

for i in range(STEP_COUNT):
    current_x = previous_x - STEP_SIZE * func_derivative(previous_x)
    previous_x = current_x

print("After", STEP_COUNT, "steps theta=", format(current_x, ".6f"), "function value=", format(func(current_x), ".6f"))
