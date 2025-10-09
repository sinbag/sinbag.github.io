import numpy as np
import matplotlib.pyplot as plt

# Parameters for the two Gaussian modes
mu1 = np.array([-2, -2])# np.array([-2, -2])  # Mean of the first Gaussian
mu2 = np.array([2, 2])  # np.array([2, 2])   # Mean of the second Gaussian
sigma1_x = 0.4 # 0.4  # Standard deviation in x for the first Gaussian
sigma1_y = 0.4 # 0.4   # Standard deviation in y for the first Gaussian
sigma2_x = 0.4 # 0.4  # Standard deviation in x for the second Gaussian
sigma2_y = 0.4 # 0.4  # Standard deviation in y for the second Gaussian
weight1 = 0.5  # Weight of the first Gaussian
weight2 = 0.5  # Weight of the second Gaussian

# Gaussian probability density function (PDF)
def gaussian_pdf(x, y, mu, sigma_x, sigma_y):
    return (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
        -0.5 * (((x - mu[0]) / sigma_x)**2 + ((y - mu[1]) / sigma_y)**2)
    )

# Score function for a single Gaussian
def score_function_2d(x, y, mu, sigma_x, sigma_y):
    dx = -(x - mu[0]) / sigma_x**2
    dy = -(y - mu[1]) / sigma_y**2
    return dx, dy

# Generate a grid of points
x = np.linspace(-4, 4, 20)
y = np.linspace(-4, 4, 20)
X, Y = np.meshgrid(x, y)

# Compute PDFs for the two Gaussians
pdf1 = gaussian_pdf(X, Y, mu1, sigma1_x, sigma1_y)
pdf2 = gaussian_pdf(X, Y, mu2, sigma2_x, sigma2_y)

target = pdf1 + pdf2

# def target(X,Y):
#     return pdf1(X,Y) + pdf2(X,Y)

# Compute the weights of the score functions
w1 = weight1 * pdf1 / (weight1 * pdf1 + weight2 * pdf2)
w2 = weight2 * pdf2 / (weight1 * pdf1 + weight2 * pdf2)

# Compute the score functions for the two Gaussians
U1, V1 = score_function_2d(X, Y, mu1, sigma1_x, sigma1_y)
U2, V2 = score_function_2d(X, Y, mu2, sigma2_x, sigma2_y)

# Combine the score functions, weighted by the respective probabilities
U = w1 * U1 + w2 * U2
V = w1 * V1 + w2 * V2

# # Plot the vector field with smaller arrows
# plt.figure(figsize=(8, 8))
# plt.quiver(X, Y, U, V, color='orange', angles='xy', scale_units='xy', scale=50, alpha=0.9)
# plt.axhline(0, color='black', linewidth=0.2, linestyle='--')
# plt.axvline(0, color='black', linewidth=0.2, linestyle='--')
# plt.title("Vector Field of the Score Function (Bimodal 2D Gaussian)", fontsize=16)
# plt.xlabel("x", fontsize=14)
# plt.ylabel("y", fontsize=14)
# plt.xlim(-4, 4)
# plt.ylim(-4, 4)
# plt.grid(alpha=0.3)
# plt.show()

bound_lim=4
fig, ax = plt.subplots()
# ax.contourf(x, y, target, levels=50, cmap='Oranges', alpha=0.5)
ax.set_xlim([-bound_lim, bound_lim])
ax.set_ylim([-bound_lim, bound_lim])
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

# ax.quiver(X, Y, U, V, color='orange', angles='xy', scale_units='xy', scale=50, alpha=0.9)
ax.contourf(X, Y, target, levels=50, cmap='Oranges', alpha=0.5)
# ax.axhline(0, color='black', linewidth=0.2, linestyle='--')
# ax.axvline(0, color='black', linewidth=0.2, linestyle='--')

plt.savefig("bimodal_gaussian.pdf",bbox_inches='tight')
plt.show()