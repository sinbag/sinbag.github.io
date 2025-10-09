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

# SGLD optimization function
def sgld_update(position, grad, lr, noise_scale):
    noise = np.random.normal(scale=noise_scale, size=position.shape)
    return position - lr * grad + np.sqrt(2*lr) * noise

# Animation setup
np.random.seed(42)
x_range = np.linspace(-2, 2, 400)
y_range = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

# Initial position
sgld_pos = np.array([-1.5, 1.5])

# Parameters
learning_rate = 0.001
noise_scale = 0.1
steps = 10000

sgld_trajectory = [sgld_pos.copy()]

# Precompute SGLD trajectory
for _ in range(steps):
    grad_sgld = rosenbrock_grad(*sgld_pos)
    sgld_pos = sgld_update(sgld_pos, grad_sgld, learning_rate, noise_scale)
    sgld_trajectory.append(sgld_pos.copy())

sgld_trajectory = np.array(sgld_trajectory)

# Plot and animate
fig, ax = plt.subplots(figsize=(8, 6))
cs = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
cbar = fig.colorbar(cs)
ax.plot([], [], 'y-', label='SGLD', lw=2)
sgld_line, = ax.plot([], [], 'white', markersize=5,lw=2)
# ax.legend()

ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

def update(frame):
    sgld_line.set_data(sgld_trajectory[:frame, 0], sgld_trajectory[:frame, 1])
    return sgld_line,

ani = FuncAnimation(fig, update, frames=range(1, steps), interval=5, blit=True)
ani.save(filename="sgd_with_noise_rosenbrock2d.mp4", writer="ffmpeg",)
plt.show()
