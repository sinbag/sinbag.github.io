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

# SGD and SGLD optimization functions
def sgd_update(position, grad, lr):
    return position - lr * grad

def sgld_update(position, grad, lr, noise_scale):
    noise = np.random.normal(scale=noise_scale, size=position.shape)
    return position - lr * grad + noise

# Animation setup
np.random.seed(42)
x_range = np.linspace(-2, 2, 400)
y_range = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

# Initial positions
sgd_pos = np.array([-1.5, 1.5])
sgld_pos = np.array([-1.5, 1.5])

# Parameters
learning_rate = 0.001
noise_scale = 0.1
steps = 1000

sgd_trajectory = [sgd_pos.copy()]
sgld_trajectory = [sgld_pos.copy()]

# Precompute trajectories
for _ in range(steps):
    grad_sgd = rosenbrock_grad(*sgd_pos)
    grad_sgld = rosenbrock_grad(*sgld_pos)
    sgd_pos = sgd_update(sgd_pos, grad_sgd, learning_rate)
    sgld_pos = sgld_update(sgld_pos, grad_sgld, learning_rate, noise_scale)
    sgd_trajectory.append(sgd_pos.copy())
    sgld_trajectory.append(sgld_pos.copy())

sgd_trajectory = np.array(sgd_trajectory)
sgld_trajectory = np.array(sgld_trajectory)

# Plot and animate
fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(X, Y, Z, levels=50, cmap='viridis')
ax.plot([], [], 'y-', label='SGLD', lw=2)
ax.plot([], [], 'r-', label='SGD', lw=2)
sgld_line, = ax.plot([], [], 'yo', markersize=5)
sgd_line, = ax.plot([], [], 'ro', markersize=5)
ax.legend()

def update(frame):
    sgld_line.set_data(sgld_trajectory[:frame, 0], sgld_trajectory[:frame, 1])
    sgd_line.set_data(sgd_trajectory[:frame, 0], sgd_trajectory[:frame, 1])
    return sgd_line, sgld_line

ani = FuncAnimation(fig, update, frames=range(1, steps), interval=50, blit=True)
plt.show()
