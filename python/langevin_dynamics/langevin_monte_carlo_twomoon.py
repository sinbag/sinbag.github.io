import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from scipy.stats import gaussian_kde

# Generate the two-moons dataset
n_samples = 10000
data, _ = make_moons(n_samples=n_samples, noise=0.1,random_state=42)

# Compute the KDE for the two-moons dataset
kde = gaussian_kde(data.T)

def target_distribution(x):
    return kde.evaluate(x.T)

def target_log_prob(x):
    return np.log(kde.evaluate(x.T))

def grad_target_log_prob(x, epsilon=1e-5):
    # Numerical gradient of the log-probability
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_step = np.copy(x)
            x_step[i, j] += epsilon
            grad[i, j] = (target_log_prob(x_step) - target_log_prob(x)) / epsilon
    return grad

def langevin_monte_carlo(num_samples, dim, step_size, burn_in):
    samples = np.zeros((num_samples, dim))
    current_sample = np.random.randn(1, dim)  # Initialize with a random sample
    
    for i in range(num_samples + burn_in):
        grad_log_prob = grad_target_log_prob(current_sample)
        noise = np.random.randn(1, dim)
        next_sample = current_sample + 0.5 * step_size * grad_log_prob + np.sqrt(step_size) * noise
        
        # Accept the next sample
        current_sample = next_sample
        
        if i >= burn_in:
            samples[i - burn_in] = current_sample
            
    return samples

def metropolis_langevin_monte_carlo(num_samples, dim, step_size, burn_in):
    samples = np.zeros((num_samples, dim))
    current_sample = np.random.randn(1, dim)  # Initialize with a random sample
    
    for i in range(num_samples + burn_in):
        grad_log_prob = grad_target_log_prob(current_sample)
        noise = np.random.randn(1, dim)
        next_sample = current_sample + 0.5 * step_size * grad_log_prob + np.sqrt(step_size) * noise
        
        #define acceptance probability
        alpha = target_distribution(next_sample) / target_distribution(current_sample)

        # Accept or reject the step
        if np.random.rand() < alpha:
            current_sample = next_sample
        
        if i >= burn_in:
            samples[i - burn_in] = current_sample
            
    return samples

# Parameters
num_samples = 10000  # Number of samples to generate
dim = 2            # Dimension of the target distribution
step_size = 0.01    # Step size for the Langevin dynamics
burn_in = 100      # Number of burn-in steps

# Generate samples
lmc_samples = langevin_monte_carlo(num_samples, dim, step_size, burn_in)
mala_samples = metropolis_langevin_monte_carlo(num_samples, dim, step_size, burn_in)


# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot samples
# axs[0].plot(positions[:, 0], positions[:, 1], lw=1)
axs[0].scatter(lmc_samples[:, 0], lmc_samples[:, 1], s=1)
axs[0].set_title('Unadjusted Langevin Monte Carlo')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_xlim(-2, 2.5)
axs[0].set_ylim(-1.5, 1.5)
axs[0].grid(True)
axs[0].set_aspect(0.75)


axs[1].scatter(mala_samples[:, 0], mala_samples[:, 1], s=1)
axs[1].set_title('Metropolis_adjusted Langevin Monte Carlo')
axs[1].set_xlabel('X1')
axs[1].set_ylabel('X2')
axs[1].set_xlim(-2, 2.5)
axs[1].set_ylim(-1.5, 1.5)
axs[1].grid(True)
axs[1].set_aspect(0.75)


plt.tight_layout()
plt.show()