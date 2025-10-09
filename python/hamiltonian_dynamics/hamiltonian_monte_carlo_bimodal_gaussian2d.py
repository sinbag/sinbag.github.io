import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define parameters for the bimodal Gaussian
mean1 = np.array([2, 2])
cov1 = np.array([[1, 0.5], [0.5, 1]])
mean2 = np.array([-2, -2])
cov2 = np.array([[1, -0.5], [-0.5, 1]])

# Calculate the bimodal distribution
def bimodal_gaussian(x):
    return 0.5 * multivariate_normal.pdf(x, mean1, cov1) + 0.5 * multivariate_normal.pdf(x, mean2, cov2)

# Potential energy is the negative log-probability
def potential_energy(x):
    return -np.log(bimodal_gaussian(x) + 1e-9)

# Gradient of the potential energy
def gradient_potential_energy(x):
    eps = 1e-5
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_eps = x.copy()
        x_eps[i] += eps
        grad[i] = (potential_energy(x_eps) - potential_energy(x)) / eps
    return grad

# Define Hamiltonian Dynamics

def kinetic_energy(p):
    return 0.5 * np.sum(p ** 2)

def hamiltonian(x, p):
    return potential_energy(x) + kinetic_energy(p)


# Implement HMC Algorithm

def hmc_sampling(initial_position, num_samples, step_size, num_steps):
    samples = []
    current_position = np.array(initial_position, dtype=np.float32)

    for _ in range(num_samples):
        current_momentum = np.random.randn(*current_position.shape)
        proposed_position = current_position.copy()
        proposed_momentum = current_momentum.copy()

        # Simulate Hamiltonian dynamics
        proposed_momentum -= 0.5 * step_size * gradient_potential_energy(proposed_position)
        for _ in range(num_steps):
            proposed_position += step_size * proposed_momentum
            proposed_momentum -= step_size * gradient_potential_energy(proposed_position)
        proposed_momentum -= 0.5 * step_size * gradient_potential_energy(proposed_position)

        # Metropolis acceptance step
        current_hamiltonian = hamiltonian(current_position, current_momentum)
        proposed_hamiltonian = hamiltonian(proposed_position, proposed_momentum)
        acceptance_prob = np.exp(current_hamiltonian - proposed_hamiltonian)
        if np.random.rand() < acceptance_prob:
            current_position = proposed_position

        samples.append(current_position.copy())

    return np.array(samples)


# Run the Algorithm and Visualize the Results

# Parameters for HMC
initial_position = [0, 0]
num_samples = 1000
step_size = 0.1
num_steps = 10

# Generate samples
samples = hmc_sampling(initial_position, num_samples, step_size, num_steps)

# Plot the samples
plt.figure(figsize=(8, 8))
x, y = np.mgrid[-5:5:.01, -5:5:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
rv1 = multivariate_normal(mean1, cov1)
rv2 = multivariate_normal(mean2, cov2)
plt.contour(x, y, 0.5 * rv1.pdf(pos) + 0.5 * rv2.pdf(pos))
plt.scatter(samples[:, 0], samples[:, 1], s=1, c='red')
plt.title('HMC Samples from Bimodal Gaussian Distribution')
plt.show()

