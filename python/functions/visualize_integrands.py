import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Set the target distribution: a bimodal 2D Gaussian mixture
# mean1 = np.array([2, 2])
# mean2 = np.array([-2, -2])
# cov = 0.8 * np.eye(2)  # Shared covariance matrix for both modes

mean_val = 5
bound_lim = 2 #2*mean_val

# modes = [
#     {"mean": np.array([0, 0]), "cov": np.array([[0.5, 0], [0, 0.5]])},
#     {"mean": np.array([mean_val, mean_val]), "cov": np.array([[0.8, 0.3], [0.3, 0.8]])},
#     {"mean": np.array([-mean_val, mean_val]), "cov": np.array([[0.6, -0.2], [-0.2, 0.6]])},
#     {"mean": np.array([mean_val, -mean_val]), "cov": np.array([[1.0, 0.5], [0.5, 1.0]])}
# ]

# modes = [
#     {"mean": np.array([0, 0]), "cov": np.array([[0.8, 0.0], [0.0, 0.8]])},
#     {"mean": np.array([mean_val, mean_val]), "cov": np.array([[0.8, 0.0], [0.0, 0.8]])},
#     {"mean": np.array([-mean_val, mean_val]), "cov": np.array([[0.8, 0.0], [0.0, 0.8]])},
#     {"mean": np.array([mean_val, -mean_val]), "cov": np.array([[0.8, 0.0], [0.0, 0.8]])},
# ]

modes = [
    {"mean": np.array([0.6, 0.6]), "cov": np.array([[0.4, 0.0], [0.0, 0.4]])}
    # {"mean": np.array([0.6, 0.6]), "cov": np.array([[0.4, 0.0], [0.0, 0.4]])}
]


# print(cov[0, 0],cov[0, 1],cov[1, 0],cov[1, 1])

# Plotting
x, y = np.mgrid[-bound_lim:bound_lim:100j, -bound_lim:bound_lim:100j]
pos = np.dstack((x, y))
# Calculate the probability density for each mode and sum them up
target = np.zeros(x.shape)
for mode in modes:
    rv = multivariate_normal(mean=mode["mean"], cov=mode["cov"])
    target += rv.pdf(pos)

# Plotting
# plt.figure(figsize=(10, 10))
# Plot the multimodal distribution as a contour or heatmap
# plt.contourf(x, y, target, levels=50, cmap='Blues', alpha=0.5)


# Plot the result
fig, ax = plt.subplots()
ax.contourf(x, y, target, levels=50, cmap='Oranges', alpha=0.5)
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

# Remove the legend
ax.legend().set_visible(False)

# plt.title("Gaussian distribution")
plt.grid()
plt.savefig("bimodal_gaussian_isotropic.pdf", bbox_inches='tight')
plt.show()