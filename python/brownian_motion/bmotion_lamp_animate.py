# Brownian motion following a 2D image as the target distribution
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from matplotlib.animation import FuncAnimation

# fix the random seed
np.random.seed(42)

# def load_image(image)
# Load the image
image = io.imread('lamp.jpeg')

# Convert the image to grayscale
image_gray = color.rgb2gray(image)

# Invert the grayscale image to have higher values where the image is darker
image_gray = 1 - image_gray

# Normalize the grayscale image to have values between 0 and 1
image_gray /= image_gray.max()

# Define the target distribution function based on the grayscale image
def target_distribution(x, y):
    # Convert x, y coordinates to pixel indices
    i = int((x + 1) * (image_gray.shape[0] - 1) / 2)
    j = int((y + 1) * (image_gray.shape[1] - 1) / 2)
    # Return the probability density at the corresponding pixel
    return image_gray[i, j]

# Random walk simulating Brownian motion
def brownian_motion(num_steps, step_size):
    # Initialize position at the origin
    position = np.zeros(2)
    positions = [position.copy()]

    # Iterate through each step
    for _ in range(num_steps):
        # Generate a proposed step
        step = np.random.normal(0, step_size, size=2)
        new_position = position + step

        # Clip the new position to stay within range [-1, 1]
        new_position = np.clip(new_position, -1, 1)

        # Calculate acceptance probability
        # alpha = target_distribution(*new_position) / target_distribution(*position)

        # Accept or reject the step
        # if np.random.rand() < alpha:
            # position = new_position

        # Store the updated position
        positions.append(position.copy())

    return np.array(positions)

# Metropolis-Hastings algorithm for simulating Brownian motion
def brownian_motion_metropolis(num_steps, step_size):
    # Initialize position at the origin
    position = np.zeros(2)
    positions = [position.copy()]

    # Iterate through each step
    for _ in range(num_steps):
        # Generate a proposed step
        step = np.random.normal(0, step_size, size=2)
        new_position = position + step

        # Clip the new position to stay within range [-1, 1]
        new_position = np.clip(new_position, -1, 1)

        # Calculate acceptance probability
        alpha = target_distribution(*new_position) / target_distribution(*position)

        # Accept or reject the step
        if np.random.rand() < alpha:
            position = new_position

        # Store the updated position
        positions.append(position.copy())

    return np.array(positions)

# Number of steps and step size
num_steps = 8000
step_size = 0.1

# Simulate Brownian motion using Metropolis-Hastings
# positions = brownian_motion_metropolis(num_steps, step_size)
positions = brownian_motion

# length
length = positions.shape[0]
print(length)


# Plot the result
fig, ax = plt.subplots()
# circle = plt.Circle((0, 0), radius, color='b', fill=False, linestyle='--')
# ax.add_artist(circle)
# ax.plot(positions[:, 1], -positions[:, 0], 'g-', alpha=0.6, label='Brownian motion')
# ax.plot(positions[0, 0], positions[0, 1], 'ro', label='Start')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
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




# plt.title('Brownian Motion Within a Circular Ring of Thickness 0.1')
# plt.savefig('bmotion_lamp.pdf',bbox_inches='tight')
plt.imshow(image, cmap='gray', extent=(-1, 1, -1, 1))
# Create animation
ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=2)
# ani.save(filename="./pillow_example.gif", writer="pillow")
ani.save(filename="./bmotion_lamp_animate.mp4", writer="ffmpeg")
plt.show()