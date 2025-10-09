import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define a simple 2D function and its gradient
def f(x, y):
    """A simple quadratic function: f(x, y) = x^2 + y^2"""
    return x**2 + y**2

def grad_f(x, y):
    """Gradient of the function: grad(f) = [df/dx, df/dy]"""
    return np.array([2*x, 2*y])

# Stochastic Gradient Descent implementation
def sgd(x0, y0, lr=0.1, steps=50):
    path = [(x0, y0)]
    x, y = x0, y0
    for _ in range(steps):
        grad = grad_f(x, y)
        x -= lr * grad[0]
        y -= lr * grad[1]
        path.append((x, y))
    return np.array(path)

# Generate data for the function surface
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Initial point and learning rate
x0, y0 = 1.8, -1.5
learning_rate = 0.1
path = sgd(x0, y0, lr=learning_rate, steps=50)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contour(X, Y, Z, levels=30, cmap="viridis")
sc = ax.scatter([], [], color="red", label="SGD Path", zorder=5)
line, = ax.plot([], [], color="blue", lw=1)

# Initialize the animation
def init():
    sc.set_offsets([])
    line.set_data([], [])
    return sc, line

# Update function for animation
def update(frame):
    print('frame :', frame)
    line.set_data(path[:frame,0], path[:frame,1])
    sc.set_data(path[frame,0], path[frame,1])
    # current_path = path[:frame+1]
    # x_vals, y_vals = zip(*current_path)
    # # Set current point (ensure 2D structure)
    # sc.set_offsets(np.array([[x_vals[-1], y_vals[-1]]]))
    # # Set line connecting path
    # line.set_data(x_vals, y_vals)
    return sc, line
print(len(path))
# Create the animation
ani = FuncAnimation(fig, update, frames=len(path), init_func=init, blit=True, interval=200)

# Labels and legend
ax.set_title("Stochastic Gradient Descent (SGD) Visualization")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

# Show the plot (or save as needed)
plt.show()
