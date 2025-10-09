import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from matplotlib.animation import FuncAnimation

# Define the target distribution (grid function)
# def target_distribution(x, y):
    # return np.sin(x)**2 + np.cos(y)**2


# Define the target distribution (circle)
def target_distribution(x, y, radius=1):
    return (x**2 + y**2 <= radius**2).astype(float)



# Load the image
image = io.imread('lamp.jpeg')

# Convert the image to grayscale
image_gray = color.rgb2gray(image)

# Invert the grayscale image to have higher values where the image is darker
image_gray = 1 - image_gray

# Normalize the grayscale image to have values between 0 and 1
image_gray /= image_gray.max()



# # Define the target distribution function based on the grayscale image
# def target_distribution(x, y):
#     # Convert x, y coordinates to pixel indices
#     i = int((x + 1) * (image_gray.shape[0] - 1) / 2)
#     j = int((y + 1) * (image_gray.shape[1] - 1) / 2)

#     # if i > 511 or j > 511:
#     # print(x, y, i, j)


#     # Return the probability density at the corresponding pixel
#     return image_gray[i, j]

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
num_steps = 10000
step_size = 0.1

# Simulate Brownian motion using Metropolis-Hastings
positions = brownian_motion_metropolis(num_steps, step_size)

# Generate grid points for target distribution
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Z = target_distribution(X, Y)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot Brownian motion trajectory
axs[0].plot(positions[:, 0], positions[:, 1], lw=1)
axs[0].set_title('Brownian Motion with Target Distribution')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_aspect(1)

line, = axs[0].plot([], [], 'b-', lw=1)
dot, = axs[0].plot([], [], 'ro')


# Initialization function
def init():
    line.set_data([], [])
    dot.set_data([], [])
    return line, dot


# Update function for animation
def update(frame):
    line.set_data(positions[:frame,0], positions[:frame,1])
    dot.set_data(positions[frame,0], positions[frame,1])
    return line, dot

# Plot target distribution
# contour = axs[1].contourf(X, Y, Z, cmap='viridis')
# plt.imshow(Z, cmap='gray', extent=(-1, 1, -1, 1))
# fig.colorbar(contour, ax=axs[1], label='Probability Density')
axs[1].set_title('Target Distribution')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')

# Create animation
ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=2)
# ani.save(filename="./pillow_example.gif", writer="pillow")
# ani.save(filename="./ffmpeg_example.mp4", writer="ffmpeg")
plt.tight_layout()
plt.show()
 