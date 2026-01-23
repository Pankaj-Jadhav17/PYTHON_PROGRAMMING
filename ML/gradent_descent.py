def f(x):
    """Objective function: sum of squared errors for three points."""
    return ((1.4 - (x + 0.64 * 0.5)) ** 2
            + (1.9 - (x + 0.64 * 2.3)) ** 2
            + (3.2 - (x + 0.64 * 2.9)) ** 2)


def df(x):
    """Derivative of f with respect to x."""
    # derivative of (y - (x + c))^2 w.r.t x is -2*(y - (x+c))
    grad = -2 * (1.4 - (x + 0.64 * 0.5))
    grad += -2 * (1.9 - (x + 0.64 * 2.3))
    grad += -2 * (3.2 - (x + 0.64 * 2.9))
    return grad


def within_tolerance(x_old, x_new, tolerance):
    return abs(x_new - x_old) < tolerance


def gradient_descent(x_old, learning_rate, max_iterations, tolerance):
    iterations_no = 0
    close_enough = False
    x_new = x_old
    while iterations_no < max_iterations and not close_enough:
        slope = df(x_old)
        x_new = x_old - learning_rate * slope
        close_enough = within_tolerance(x_old, x_new, tolerance)
        x_old = x_new
        iterations_no += 1

    print("Minimum of the given function is at", x_new)
    print("f(min) =", f(x_new))
    return x_new, iterations_no


if __name__ == "__main__":
    gradient_descent(x_old=0.0, learning_rate=0.01, max_iterations=1000, tolerance=1e-6)