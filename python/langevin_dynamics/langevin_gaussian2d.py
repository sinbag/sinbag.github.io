import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Set parameters for the 2D Gaussian target distribution
mean = np.array([0, 0])   # Mean of the target Gaussian
cov = np.array([[1, 0.5], [0.5, 1]])  # Covariance matrix of the target Gaussian

# Define the gradient of the log-density of the target distribution
def grad_log_prob(x):
    inv_cov = np.linalg.inv(cov)
    return -np.dot(inv_cov, (x - mean))

# Langevin Monte Carlo (LMC) sampling function
def langevin_monte_carlo(num_samples, step_size, burn_in):
    samples = []
    x = np.random.randn(2)  # Initial position
    
    for _ in range(num_samples + burn_in):
        # Update step based on the LMC update rule
        x += step_size * grad_log_prob(x) + np.sqrt(2 * step_size) * np.random.randn(2)
        
        # Discard burn-in samples
        if _ >= burn_in:
            samples.append(x.copy())
    
    return np.array(samples)

# Simulation parameters
num_samples = 5000
step_size = 0.1
burn_in = 1000

# Run the Langevin Monte Carlo simulation
samples = langevin_monte_carlo(num_samples, step_size, burn_in)

# Plot the target distribution and samples
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x, y))
rv = multivariate_normal(mean, cov)

plt.figure(figsize=(8, 6))
# Plot the contour of the target Gaussian distribution
plt.contour(x, y, rv.pdf(pos), levels=10, cmap="Blues")

# Overlay the LMC samples
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, color="red", s=1)
plt.title("2D Gaussian Distribution with LMC Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# # Define parameters
# mean = np.array([0.0, 0.0])  # Mean of the 2D Gaussian
# cov = np.eye(2)  # Covariance matrix (identity matrix for simplicity)
# num_samples = 1000  # Number of samples to generate
# step_size = 0.1  # Step size (epsilon in ULMC)
# burn_in = 100  # Number of burn-in steps to discard

# # Gradient of the log-probability for a Gaussian N(0, I)
# def gradient_log_prob(x):
#     return -x  # For N(0, I), the gradient is simply -x

# # ULMC sampling function
# def ulmc_sampler(start, num_samples, step_size, burn_in):
#     samples = []
#     x = start  # Initialize starting point

#     for i in range(num_samples + burn_in):
#         # Generate noise from a standard normal distribution
#         noise = np.random.normal(size=2)
        
#         # Update x using ULMC update rule
#         x = x + 0.5 * step_size * gradient_log_prob(x) + np.sqrt(step_size) * noise
        
#         # Collect samples after burn-in
#         if i >= burn_in:
#             samples.append(x)
    
#     return np.array(samples)

# # Run the ULMC sampler
# start = np.array([2.0, 2.0])  # Starting point
# samples = ulmc_sampler(start, num_samples, step_size, burn_in)

# # Plot the samples
# plt.figure(figsize=(6, 6))
# plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10, color="blue", label="Samples")
# plt.title("Unadjusted Langevin Monte Carlo (ULMC) for 2D Gaussian")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.axis('equal')
# plt.show()
