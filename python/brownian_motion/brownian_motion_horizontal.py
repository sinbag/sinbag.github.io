import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
num_steps = 100       # Number of steps in the random walk
delta_t = 0.1          # Time step (increases smoothness in movement)
step_size = 1.0        # Step size (affects speed of movement)

# Initialize position arrays
x = np.zeros(num_steps)
y = np.zeros(num_steps)
sigma = 0.001
# Generate random steps in x and y
for i in range(1, num_steps):
    x[i] = i
    y[i] = y[i - 1] + step_size * np.random.normal(0, sigma, 1)
    sigma += 0.001

# Create plot
fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(-2, 2)
ax.plot(x, y, 'b-' ,lw=2)
line, = ax.plot([], [], 'b-', lw=2)
dot, = ax.plot([], [], 'ro')

# Initialization function
def init():
    line.set_data([], [])
    dot.set_data([], [])
    return line, dot

# Update function for animation
def update(frame):
    line.set_data(x[:frame], y[:frame])
    # dot.set_data(x[:frame], y[:frame])
    return line, dot

# Hide axes, ticks, and labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Create animation
# ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=100)
# ani.save(filename="./pillow_example.gif", writer="pillow")
# ani.save(filename="./ffmpeg_example.mp4", writer="ffmpeg")
# ani.save(filename='./brownian_motion_data_to_prior.gif',writer='pillow',savefig_kwargs={"transparent": True})
plt.savefig('bmotion_horizontal.png', bbox_inches='tight', transparent=True, dpi=300)
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
