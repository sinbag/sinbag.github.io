import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.animation import FuncAnimation

mean_val = 5
bound_lim = 2*mean_val

modes = [
    {"mean": np.array([0, 0]), "cov": np.array([[0.5, 0], [0, 0.5]])},
    {"mean": np.array([mean_val, mean_val]), "cov": np.array([[0.8, 0.3], [0.3, 0.8]])},
    {"mean": np.array([-mean_val, mean_val]), "cov": np.array([[0.6, -0.2], [-0.2, 0.6]])},
    {"mean": np.array([mean_val, -mean_val]), "cov": np.array([[1.0, 0.5], [0.5, 1.0]])}
]

# modes = [
#     {"mean": np.array([0, 0]), "cov": np.array([[0.8, 0.0], [0.0, 0.8]])},
#     {"mean": np.array([mean_val, mean_val]), "cov": np.array([[0.8, 0.0], [0.0, 0.8]])},
#     {"mean": np.array([-mean_val, mean_val]), "cov": np.array([[0.8, 0.0], [0.0, 0.8]])},
#     {"mean": np.array([mean_val, -mean_val]), "cov": np.array([[0.8, 0.0], [0.0, 0.8]])},
# ]

# print(modes[2]["cov"].shape)


# Gradient of the log-probability density function
def grad_log_density(position):
    grad = np.zeros(2)
    total_density = sum(multivariate_normal(mean=mode["mean"], cov=mode["cov"]).pdf(position) for mode in modes)
    
    for mode in modes:
        mean = mode["mean"]
        cov = mode["cov"]
        cov_inv = np.linalg.inv(mode["cov"])
        density = multivariate_normal(mean=mean, cov=mode["cov"]).pdf(position)
        grad += np.dot(cov_inv, -(position - mean)) * density
    
    if total_density:
        return grad / total_density
    else:
        print(grad, cov_inv, total_density, position)
        return 0.1



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
step_size = 1
start_point = np.array([0.0, 0.0])

# Run the sampler
samples = langevin_monte_carlo(n_samples, step_size, start_point)

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

# To animate the brownian motion
line, = ax.plot([], [], 'y-', label='Brownian motion')
dot, = ax.plot([], [], 'ro')
# dot, = ax.scatter(positions[:, 0], positions[:, 1], color = 'green', s=10)


# Initialization function
def init():
    line.set_data([], [])
    dot.set_data([], [])
    return line, dot


# Update function for animation
def update(frame):
    line.set_data(samples[:frame,0], samples[:frame,1])
    dot.set_data(samples[frame,0], samples[frame,1])
    return line, dot



# Create animation
ani = FuncAnimation(fig, update, frames=n_samples, init_func=init, blit=True, interval=2)
ani.save(filename="langevin_multimodal_gaussian_anisotropic.mp4", writer="ffmpeg")


# Plot the ULA samples
# ax.scatter(samples[:, 0], samples[:, 1], c=np.arange(len(samples)), cmap='viridis', s=5)
# plt.colorbar(label="Sample Index")
# plt.title("Samples from Multimodal Gaussian Distribution Using ULA with Domain Constraints")
# plt.xlabel("X position")
# plt.ylabel("Y position")
# plt.grid()
plt.savefig('mala_multimodal.pdf',bbox_inches='tight')
plt.show()