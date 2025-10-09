import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the 2D function and its gradient
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def log_rosenbrock(x,y):
    return np.log(rosenbrock(x,y))

def rosenbrock_grad(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

def grad_log_rosenbrock(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])/rosenbrock(x, y)


# SGLD optimization function
def sgld_update(position, grad, lr, temperature=1):
    # noise = np.random.normal(scale=np.sqrt(2 * lr * temperature), size=position.shape)
    noise = np.random.normal(scale=0.1, size=position.shape)
    return position - lr * grad + np.sqrt(2* lr) * noise

# Animation setup
np.random.seed(42)
x_range = np.linspace(-2, 2, 400)
y_range = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

# Initial position
sgld_pos = np.array([-1.5, 1.5])

# Parameters
step_size = 0.01
temperature = 1.0  # Controls the scale of the noise
steps = 10000

sgld_trajectory = [sgld_pos.copy()]

# Precompute SGLD trajectory
for _ in range(steps):
    # grad_sgld = rosenbrock_grad(*sgld_pos)
    grad_sgld = grad_log_rosenbrock(*sgld_pos)
    sgld_pos = sgld_update(sgld_pos, grad_sgld, step_size, temperature)
    sgld_trajectory.append(sgld_pos.copy())

sgld_trajectory = np.array(sgld_trajectory)

# Plot and animate
fig, ax = plt.subplots(figsize=(8, 6))
cs = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
# ax.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar = fig.colorbar(cs)
ax.plot([], [], 'y-', label='SGLD', lw=2)
sgld_line, = ax.plot([], [], 'y-', markersize=5, lw=2)
ax.legend()

def update(frame):
    sgld_line.set_data(sgld_trajectory[:frame, 0], sgld_trajectory[:frame, 1])
    return sgld_line,

ani = FuncAnimation(fig, update, frames=range(1, steps), interval=5, blit=True)
ani.save(filename="sgld_rosenbrock2d.mp4", writer="ffmpeg",)
plt.show()
