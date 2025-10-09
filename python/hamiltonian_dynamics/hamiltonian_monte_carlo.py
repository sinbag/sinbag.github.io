import numpy as np
from scipy.stats import norm

def potential_energy(x):
    """Potential energy for a standard Gaussian distribution."""
    return 0.5 * x**2

def gradient_potential_energy(x):
    """Gradient of the potential energy for a standard Gaussian distribution."""
    return x

def hamiltonian_monte_carlo(n_samples, initial_position, step_size, n_steps):
    """Perform Hamiltonian Monte Carlo sampling."""
    samples = [initial_position]
    current_position = initial_position

    for _ in range(n_samples):
        position = current_position
        momentum = np.random.normal(0, 1)
        initial_momentum = momentum

        # Simulate Hamiltonian dynamics
        for _ in range(n_steps):
            momentum -= 0.5 * step_size * gradient_potential_energy(position)
            position += step_size * momentum
            momentum -= 0.5 * step_size * gradient_potential_energy(position)

        # Metropolis acceptance step
        current_potential = potential_energy(current_position)
        current_kinetic = 0.5 * initial_momentum**2
        proposed_potential = potential_energy(position)
        proposed_kinetic = 0.5 * momentum**2

        if np.random.rand() < np.exp(current_potential + current_kinetic - proposed_potential - proposed_kinetic):
            current_position = position

        samples.append(current_position)

    return np.array(samples)

# Parameters
n_samples = 1000
initial_position = 0.0
step_size = 0.1
n_steps = 10

# Run HMC
samples = hamiltonian_monte_carlo(n_samples, initial_position, step_size, n_steps)

# Plot results
import matplotlib.pyplot as plt

plt.hist(samples, bins=30, density=True, label='HMC samples')
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x), label='True distribution')
plt.legend()
plt.show()
