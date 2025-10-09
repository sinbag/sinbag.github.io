# Brownian motion with 4 different target distribution
# to be added after acceptance

# Brownian motion for a circle distribution

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
num_steps = 10000   # Number of time steps
radius = 5.0        # Radius of the circle
delta_t = 0.01      # Time step size
x, y = 0.0, 0.0    # Starting at the origin (can be any point within the circle)

# List to store positions
positions = [(x, y)]

# Function to reflect point if outside the circle
def reflect(x, y, radius):
    distance = np.sqrt(x**2 + y**2)
    if distance > radius:
        # Scale back the point to lie on the circle's circumference
        x = x / distance * radius
        y = y / distance * radius
    return x, y

# Simulate Brownian motion
for i in range(num_steps):
    # Generate random movement
    dx, dy = np.random.normal(scale=np.sqrt(delta_t), size=2)
    
    # Update position
    x += dx
    y += dy
    
    # Reflect if the point is outside the circle
    x, y = reflect(x, y, radius)
    
    # Store the position
    positions.append((x, y))

# Convert to a numpy array for easy plotting
positions = np.array(positions)

# Plot the result
fig, ax = plt.subplots()
circle = plt.Circle((0, 0), radius, color='b', fill=False, linestyle='--')
ax.add_artist(circle)
# ax.plot(positions[:, 0], positions[:, 1], 'g-', alpha=0.6, label='Brownian motion')
# ax.plot(positions[0, 0], positions[0, 1], 'ro', label='Start')
ax.set_xlim([-radius - 0.1, radius + 0.1])
ax.set_ylim([-radius - 0.1, radius + 0.1])
ax.set_aspect('equal')


# Hide axes, ticks, and labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Remove the legend
ax.legend().set_visible(False)


# Animate the plot
line, = ax.plot([], [], 'y-', label='Brownian motion')
dot, = ax.plot([], [], 'ro')


# Initialization function
def init():
    line.set_data([], [])
    dot.set_data([], [])
    return line, dot


# Update function for animation
def update(frame):
    line.set_data(positions[:frame,1], -positions[:frame,0])
    dot.set_data(positions[frame,1], -positions[frame,0])
    return line, dot

# plt.title('Brownian Motion in a square')
# plt.savefig('bmotion_square.pdf',bbox_inches='tight')
# Create animation
ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=2)
# ani.save(filename="./pillow_example.gif", writer="pillow")
ani.save(filename="./bmotion_random_walk_disk_animate.mp4", writer="ffmpeg")
# plt.legend()
plt.show()