import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Load the image and convert it to grayscale
image_path = 'path_to_your_image.png'  # Change this to the path of your image
image = Image.open(image_path).convert('L')
image_array = np.array(image)

# Normalize the image to get a probability distribution
image_array = image_array.astype(np.float32)
image_array /= image_array.sum()

# Log the target distribution for numerical stability
log_target = np.log(image_array + 1e-9)




# Define the Hamiltonian dynamics, including the potential and kinetic energy functions:

def potential_energy(x):
    # Ensure indices are within bounds
    x = np.clip(x, [0, 0], [image_array.shape[0] - 1, image_array.shape[1] - 1])
    x = x.astype(int)
    return -log_target[x[0], x[1]]

def kinetic_energy(p):
    # Kinetic energy is quadratic in momentum
    return 0.5 * np.sum(p ** 2)

def hamiltonian(x, p):
    return potential_energy(x) + kinetic_energy(p)

def gradient_potential_energy(x):
    # Numerical gradient of the potential energy
    eps = 1e-5
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_eps = x.copy()
        x_eps[i] += eps
        grad[i] = (potential_energy(x_eps) - potential_energy(x)) / eps
    return grad


# Implement the HMC algorithm:

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
            # Reflecting boundaries
            proposed_position = np.clip(proposed_position, [0, 0], [image_array.shape[0] - 1, image_array.shape[1] - 1])
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



# Run the algorithm and visualize the results

# Parameters for HMC
initial_position = [image_array.shape[0] // 2, image_array.shape[1] // 2]
num_samples = 1000
step_size = 0.1
num_steps = 10

# Generate samples
samples = hmc_sampling(initial_position, num_samples, step_size, num_steps)

# Plot the samples
plt.figure(figsize=(8, 8))
plt.imshow(image_array, cmap='gray', origin='lower')
plt.scatter(samples[:, 1], samples[:, 0], s=1, c='red')
plt.title('HMC Samples on Target Distribution')
plt.show()


