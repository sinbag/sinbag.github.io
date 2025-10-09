import numpy as np
import matplotlib.pyplot as plt

# Parameters for the 2D Gaussian
mu = np.array([0, 0])  # Mean (mu_x, mu_y)
sigma_x = 1  # Standard deviation in x
sigma_y = 1  # Standard deviation in y

# Define the score function for the 2D Gaussian
def score_function_2d(x, y, mu, sigma_x, sigma_y):
    dx = -(x - mu[0]) / sigma_x**2
    dy = -(y - mu[1]) / sigma_y**2
    return dx, dy

# Generate a grid of points
x = np.linspace(-3 * sigma_x, 3 * sigma_x, 20)
y = np.linspace(-3 * sigma_y, 3 * sigma_y, 20)
X, Y = np.meshgrid(x, y)

# Compute the score function at each point on the grid
U, V = score_function_2d(X, Y, mu, sigma_x, sigma_y)

# Plot the vector field with smaller arrows
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, color='blue', angles='xy', scale_units='xy', scale=15, alpha=0.8)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.title("Vector Field of the Score Function (2D Gaussian)", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.xlim(-3 * sigma_x, 3 * sigma_x)
plt.ylim(-3 * sigma_y, 3 * sigma_y)
plt.grid(alpha=0.3)
plt.show()
