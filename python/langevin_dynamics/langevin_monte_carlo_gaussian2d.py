import numpy as np
import matplotlib.pyplot as plt

def target_log_prob(x, mean, cov_inv):
    """
    Log-probability of a multivariate Gaussian distribution.
    """
    diff = x - mean
    return -0.5 * np.dot(diff.T, np.dot(cov_inv, diff))

def grad_target_log_prob(x, mean, cov_inv):
    """
    Gradient of the log-probability of a multivariate Gaussian distribution.
    """
    diff = x - mean
    return -np.dot(cov_inv, diff)

def langevin_monte_carlo(num_samples, dim, step_size, burn_in, mean, cov):
    """
    Langevin Monte Carlo sampling.
    """
    samples = np.zeros((num_samples, dim))
    current_sample = np.random.randn(dim) + mean  # Initialize with a random sample around the mean
    cov_inv = np.linalg.inv(cov)  # Precompute the inverse of the covariance matrix
    
    for i in range(num_samples + burn_in):
        grad_log_prob = grad_target_log_prob(current_sample, mean, cov_inv)
        noise = np.random.randn(dim)
        next_sample = current_sample + 0.5 * step_size * grad_log_prob + np.sqrt(step_size) * noise
        
        # Accept the next sample
        current_sample = next_sample
        
        if i >= burn_in:
            samples[i - burn_in] = current_sample
            
    return samples

# Parameters
num_samples = 10000  # Number of samples to generate
dim = 2            # Dimension of the target distribution
step_size = 0.1    # Step size for the Langevin dynamics
burn_in = 100      # Number of burn-in steps
mean = np.array([0, 0])  # Mean of the target Gaussian distribution
cov = np.array([[1, 0.8], [0.8, 1]])  # Covariance matrix of the target Gaussian distribution

# Generate samples
samples = langevin_monte_carlo(num_samples, dim, step_size, burn_in, mean, cov)

# Output some of the samples
print(samples[:10])

# Plot the samples
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.title('Samples from Langevin Monte Carlo')
plt.xlabel('X1')
plt.ylabel('X2')
plt.axis('equal')
plt.show()
