import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
num_steps = 10000       # Number of steps in the random walk
delta_t = 0.1          # Time step (increases smoothness in movement)
step_size = 1.0        # Step size (affects speed of movement)

# Initialize position arrays
x = np.zeros(num_steps)
y = np.zeros(num_steps)

# Generate random steps in x and y
for i in range(1, num_steps):
    angle = 2 * np.pi * np.random.rand()  # Random direction
    x[i] = x[i - 1] + step_size * np.cos(angle)
    y[i] = y[i - 1] + step_size * np.sin(angle)

# Create plot
fig, ax = plt.subplots()
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
line, = ax.plot([], [], 'b-', lw=1)
dot, = ax.plot([], [], 'ro')

# Initialization function
def init():
    line.set_data([], [])
    dot.set_data([], [])
    return line, dot

# Update function for animation
def update(frame):
    line.set_data(x[:frame], y[:frame])
    dot.set_data(x[frame], y[frame])
    return line, dot

# Create animation
ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=2)
# ani.save(filename="./pillow_example.gif", writer="pillow")
ani.save(filename="./ffmpeg_example.mp4", writer="ffmpeg")
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Parameters
# num_steps = 1000        # Number of steps in the random walk
# step_size = 1.0         # Step size (affects speed of movement)

# # Initialize position arrays for x and y coordinates
# x = np.zeros(num_steps)
# y = np.zeros(num_steps)

# # Generate random steps in x and y for Brownian motion
# for i in range(1, num_steps):
#     angle = 2 * np.pi * np.random.rand()  # Random direction
#     x[i] = x[i - 1] + step_size * np.cos(angle)
#     y[i] = y[i - 1] + step_size * np.sin(angle)

# # Determine plot limits based on path data
# padding = 5
# x_min, x_max = np.min(x) - padding, np.max(x) + padding
# y_min, y_max = np.min(y) - padding, np.max(y) + padding

# # Create plot
# fig, ax = plt.subplots()
# ax.set_xlim(x_min, x_max)
# ax.set_ylim(y_min, y_max)
# line, = ax.plot([], [], 'b-', lw=1)  # Line representing the Brownian path
# dot, = ax.plot([], [], 'ro')         # Red dot representing the current position

# # Initialization function for the animation
# def init():
#     line.set_data([], [])
#     dot.set_data([], [])
#     return line, dot

# # Update function for the animation
# def update(frame):
#     print(x.shape, y.shape)
#     line.set_data(x[:frame], y[:frame])   # Update line path
#     dot.set_data(x[frame], y[frame])      # Update dot position
#     return line, dot

# # Create the animation
# ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=10)

# plt.show()
