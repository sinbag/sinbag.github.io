import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

mean_val = 2.5
bound_lim = 2*mean_val

# Set the target distribution: a bimodal 2D Gaussian mixture
mean1 = np.array([mean_val, mean_val])
mean2 = np.array([-mean_val, -mean_val])
cov = 0.8 * np.eye(2)  # Shared covariance matrix for both modes

# Target density function (bimodal Gaussian mixture)
def target_density(x):
    return 0.5 * multivariate_normal.pdf(x, mean=mean1, cov=cov) + 0.5 * multivariate_normal.pdf(x, mean=mean2, cov=cov)

# Gradient of the log-density of the target distribution
def grad_log_density(x):
    grad1 = -(x - mean1) / cov[0, 0]
    grad2 = -(x - mean2) / cov[0, 0]
    density1 = multivariate_normal.pdf(x, mean=mean1, cov=cov)
    density2 = multivariate_normal.pdf(x, mean=mean2, cov=cov)
    return (0.5 * density1 * grad1 + 0.5 * density2 * grad2) / target_density(x)

# Langevin Monte Carlo sampling function
def langevin_monte_carlo(n_samples, step_size, start_point):
    samples = [start_point]
    for _ in range(n_samples - 1):
        x = samples[-1]
        grad = grad_log_density(x)
        proposal = x + 0.5 * step_size * grad + np.sqrt(step_size) * np.random.normal(size=2)
        samples.append(proposal)
    return np.array(samples)

# Parameters
n_samples = 10000
step_size = 0.1
start_point = np.array([0.0, 0.0])

# Run the sampler
samples = langevin_monte_carlo(n_samples, step_size, start_point)

# Plotting
x, y = np.mgrid[-bound_lim:bound_lim:100j, -bound_lim:bound_lim:100j]
pos = np.dstack((x, y))
target = 0.5 * multivariate_normal(mean=mean1, cov=cov).pdf(pos) + 0.5 * multivariate_normal(mean=mean2, cov=cov).pdf(pos)

# Plotting
# plt.figure(figsize=(10, 10))
# Plot the multimodal distribution as a contour or heatmap
# plt.contourf(x, y, target, levels=50, cmap='Blues', alpha=0.5)
# Plot the result
fig, ax = plt.subplots()
ax.contourf(x, y, target, levels=50, cmap='Blues', alpha=0.5)
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

# Plot the ULA samples
ax.scatter(samples[:, 0], samples[:, 1], c=np.arange(len(samples)), cmap='viridis', s=5)
# plt.colorbar(label="Sample Index")
# plt.title("Samples from Multimodal Gaussian Distribution Using ULA with Domain Constraints")
# plt.xlabel("X position")
# plt.ylabel("Y position")
# plt.grid()
plt.show()