import numpy as np
import matplotlib.pyplot as plt

def target_log_prob(x):
    # Example target distribution: standard normal distribution
    return -0.5 * np.sum(x**2)

def grad_target_log_prob(x):
    # Gradient of the log of the target distribution: standard normal distribution
    return -x

def langevin_monte_carlo(num_samples, dim, step_size, burn_in):
    samples = np.zeros((num_samples, dim))
    current_sample = np.random.randn(dim)  # Initialize with a random sample
    
    for i in range(num_samples + burn_in):
        grad_log_prob = grad_target_log_prob(current_sample)
        noise = np.random.randn(dim)
        next_sample = current_sample + 0.5 * step_size * grad_log_prob + np.sqrt(step_size) * noise
        
        # Accept the next sample
        current_sample = next_sample
        
        if i >= burn_in:
            samples[i - burn_in] = current_sample
            
    return samples

# Parameters
num_samples = 1000  # Number of samples to generate
dim = 2            # Dimension of the target distribution
step_size = 0.1    # Step size for the Langevin dynamics
burn_in = 100      # Number of burn-in steps

# Generate samples
samples = langevin_monte_carlo(num_samples, dim, step_size, burn_in)

# Output some of the samples
print(samples[:10])

# Create subplots
fig, axs = plt.subplots(1, 1, figsize=(5, 5))

# Plot Brownian motion trajectory
# axs[0].plot(positions[:, 0], positions[:, 1], lw=1)
axs.scatter(samples[:, 0], samples[:, 1], s=0.5)
axs.set_title('LMC')
axs.set_xlabel('X')
axs.set_ylabel('Y')
axs.set_xlim(-3, 3)
axs.set_ylim(-3, 3)
axs.grid(True)
axs.set_aspect(1)

# # Plot the image
# axs[1].imshow(image_gray, cmap='gray')
# axs[1].set_title('Image')
# axs[1].axis('off')

plt.tight_layout()
plt.show()