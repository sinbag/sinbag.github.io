import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

# def load_image(image)
# Load the image
image = io.imread('pixar2.jpeg')

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
positions = brownian_motion_metropolis(num_steps, step_size)

# length
length = positions.shape[0]
print(length)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot Brownian motion trajectory
axs[0].plot(positions[:, 1], positions[:, 0], lw=1)
# axs[0].scatter(positions[:, 1], -positions[:, 0], s=0.5)
axs[0].set_title('Brownian Motion')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_xlim(-1, 1)
axs[0].set_ylim(-1, 1)
# axs[0].grid(True)
axs[0].set_aspect(1)

# Plot the image
axs[1].imshow(image_gray, cmap='gray')
axs[1].set_title('Image')
axs[1].axis('off')

plt.tight_layout()
plt.show()
