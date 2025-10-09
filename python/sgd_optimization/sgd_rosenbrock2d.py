import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the 2D function and its gradient
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# SGD optimization function
def sgd_update(position, grad, lr):
    return position - lr * grad

# Animation setup
np.random.seed(42)
x_range = np.linspace(-2, 2, 400)
y_range = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

# Initial position
sgd_pos = np.array([-1.5, 1.5])

# Parameters
learning_rate = 0.001
steps = 10000
bound_lim = 3

sgd_trajectory = [sgd_pos.copy()]

# Precompute SGD trajectory
for _ in range(steps):
    grad_sgd = rosenbrock_grad(*sgd_pos)
    sgd_pos = sgd_update(sgd_pos, grad_sgd, learning_rate)
    sgd_trajectory.append(sgd_pos.copy())

sgd_trajectory = np.array(sgd_trajectory)

# Plot and animate
fig, ax = plt.subplots(figsize=(8, 6))
cs = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
cbar = fig.colorbar(cs)
ax.plot([], [], 'r-', label='SGD', lw=2)
sgd_line, = ax.plot([], [], 'green', markersize=10, lw=6)
# ax.legend()

# fig, ax = plt.subplots()
# ax.contourf(x, y, target, levels=50, cmap='Oranges', alpha=0.5)
# ax.set_xlim([-bound_lim, bound_lim])
# ax.set_ylim([-bound_lim, bound_lim])
# ax.set_aspect('equal')
# Hide axes, ticks, and labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

def update(frame):
    sgd_line.set_data(sgd_trajectory[:frame, 0], sgd_trajectory[:frame, 1])
    return sgd_line,

ani = FuncAnimation(fig, update, frames=range(1, steps), interval=5, blit=True)
ani.save(filename="sgd_rosenbrock2d.mp4", writer="ffmpeg")
plt.show()
