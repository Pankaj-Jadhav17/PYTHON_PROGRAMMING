# import numpy as np
# import matplotlib.pyplot as plt

# # Function
# def f(x):
#     return (x + 5)**2

# # Derivative (slope)
# def df(x):
#     return 2 * (x + 5)

# # Generate x values for plotting the curve
# x_vals = np.linspace(-15, 5, 200)
# y_vals = f(x_vals)

# # 1. Start from a random (initial) point
# x_0 = 5
# learning_rate = 0.05
# iterations = 20

# # Store steps for plotting
# x_steps = [x_0]
# y_steps = [f(x_0)]

# # Gradient Descent Loop
# for i in range(iterations):
#     slope = df(x_0)                      # Step 2: compute slope
#     x_new = x_0 - learning_rate * slope  # Step 3: descend
#     x_steps.append(x_new)
#     y_steps.append(f(x_new))
#     x_0 = x_new

# # Plot the function
# plt.plot(x_vals, y_vals, label="f(x) = (x + 5)^2")

# # Plot gradient descent steps
# plt.scatter(x_steps, y_steps, label="Gradient Descent Steps")
# plt.plot(x_steps, y_steps)

# # Labels and title
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.title("Gradient Descent to Find the Minimum of f(x) = (x + 5)^2")
# plt.legend()
# plt.grid(True)
# plt.show()




import numpy as np
import matplotlib.pyplot as plt
# Function
def f(x):
    return -(x-1)^2+10

# Derivative (slope)
def df(x):
    return 2 * (x + 5)

# Generate x values for plotting the curve
x_vals = np.linspace(-15, 5, 200)
y_vals = f(x_vals)

# 1. Start from a random (initial) point
x_0 = 5
learning_rate = 0.05
iterations = 20

# Store steps for plotting
x_steps = [x_0]
y_steps = [f(x_0)]

# Gradient Descent Loop
for i in range(iterations):
    slope = df(x_0)                      # Step 2: compute slope
    x_new = x_0 - learning_rate * slope  # Step 3: descend
    x_steps.append(x_new)
    y_steps.append(f(x_new))
    x_0 = x_new

# Plot the function
plt.plot(x_vals, y_vals, label="f(x) = -(x-1)^2 + 10")

# Plot gradient descent steps
plt.scatter(x_steps, y_steps, label="Gradient Descent Steps")
plt.plot(x_steps, y_steps)

# Labels and title
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient Descent to Find the Minimum of f(x) = -(x-1)^2 + 10")
plt.legend()
plt.grid(True)
plt.show()

