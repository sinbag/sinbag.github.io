import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.animation import FuncAnimation

mean_val = 5
bound_lim = 2*mean_val

# Set the target distribution: a bimodal 2D Gaussian mixture
mean1 = np.array([mean_val, mean_val])
mean2 = np.array([0, 0])
mean3 = np.array([mean_val, -mean_val])
mean4 = np.array([-mean_val, mean_val])
# cov = 0.8 * np.eye(2)  # Shared covariance matrix for both modes
modes = [
    {"mean": np.array([0, 0]), "cov": np.array([[0.5, 0], [0, 0.5]])},
    {"mean": np.array([5, 5]), "cov": np.array([[0.8, 0.3], [0.3, 0.8]])},
    {"mean": np.array([-5, 5]), "cov": np.array([[0.6, -0.2], [-0.2, 0.6]])},
    {"mean": np.array([5, -5]), "cov": np.array([[1.0, 0.5], [0.5, 1.0]])}
]

detcov1 = np.sqrt(np.linalg.det(modes[0]["cov"]))    
detcov2 = np.sqrt(np.linalg.det(modes[1]["cov"]))    
detcov3 = np.sqrt(np.linalg.det(modes[2]["cov"]))    
detcov4 = np.sqrt(np.linalg.det(modes[3]["cov"]))    

# Target density function (bimodal Gaussian mixture)
def target_density(x):
    return sum(0.25 * multivariate_normal.pdf(x, mean=mode["mean"], cov=mode["cov"]) for mode in modes)

# def target_density(x):
#     return (0.25 * multivariate_normal.pdf(x, mean=mean1, cov=cov) + 
#             0.25 * multivariate_normal.pdf(x, mean=mean2, cov=cov) + 
#             0.25 * multivariate_normal.pdf(x, mean=mean3, cov=cov) + 
#             0.25 * multivariate_normal.pdf(x, mean=mean4, cov=cov))

# Gradient of the log-density of the target distribution
def grad_log_density(x):
    grad1 = -(x - mean1) / detcov1
    grad2 = -(x - mean2) / detcov2
    grad3 = -(x - mean3) / detcov3
    grad4 = -(x - mean4) / detcov4
    density1 = multivariate_normal.pdf(x, mean=mean1, cov=detcov1)
    density2 = multivariate_normal.pdf(x, mean=mean2, cov=detcov2)
    density3 = multivariate_normal.pdf(x, mean=mean3, cov=detcov3)
    density4 = multivariate_normal.pdf(x, mean=mean4, cov=detcov4)
    return (0.25 * density1 * grad1 + 
            0.25 * density2 * grad2 + 
            0.25 * density3 * grad3 + 
            0.25 * density4 * grad4) / target_density(x)

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
step_size = 0.4
start_point = np.array([0.0, 0.0])

# Run the sampler
samples = langevin_monte_carlo(n_samples, step_size, start_point)

# Plotting
x, y = np.mgrid[-bound_lim:bound_lim:100j, -bound_lim:bound_lim:100j]
pos = np.dstack((x, y))
target = target_density(pos)
# target = 0.25 * multivariate_normal(mean=mean1, cov=cov).pdf(pos) + 0.25 * multivariate_normal(mean=mean2, cov=cov).pdf(pos) + 0.25 * multivariate_normal(mean=mean3, cov=cov).pdf(pos) + 0.25 * multivariate_normal(mean=mean4, cov=cov).pdf(pos)

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
ax.scatter(samples[:, 0], samples[:, 1], color = 'green', s=10, alpha=1)


# Hide axes, ticks, and labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# To animate the brownian motion
line, = ax.plot([], [], 'y-', label='Brownian motion')
dot, = ax.plot([], [], 'go', markersize=2)
# dot, = ax.scatter([], [], c=np.arange(len(samples)), cmap='viridis', s=5)
# dot, = ax.scatter(positions[:, 0], positions[:, 1], color = 'green', s=10)


# Initialization function
def init():
    # line.set_data([], [])
    dot.set_data([], [])
    return line, dot


# Update function for animation
def update(frame):
    # line.set_data(samples[:frame,0], samples[:frame,1])
    dot.set_data(samples[:frame,0], samples[:frame,1])
    return line, dot



# Create animation
ani = FuncAnimation(fig, update, frames=len(samples), init_func=init, blit=True, interval=5)
# ani.save(filename="langevin_multimodal_gaussian_isotropic.mp4", writer="ffmpeg")


# Plot the ULA samples
# ax.scatter(samples[:, 0], samples[:, 1], c=np.arange(len(samples)), cmap='viridis', s=5)
# plt.savefig("langevin_multimodal_gaussian_isotropic_step_pt1.pdf",bbox_inches='tight')
# plt.colorbar(label="Sample Index")
# plt.title("Samples from Multimodal Gaussian Distribution Using ULA with Domain Constraints")
# plt.xlabel("X position")
# plt.ylabel("Y position")
# plt.grid()
plt.savefig('mala_multimodal_gaussian.pdf',bbox_inches='tight')
plt.show()