import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Parameters
num_steps = 1000
burn_in_steps = 200  # Number of steps to discard for burn-in
step_size = 0.1  # Step size for Langevin update
num_chains = 5  # Number of independent chains to run
domain_min, domain_max = -10, 10  # Domain boundaries

# Define the centers and covariances of each Gaussian mode in the target distribution
modes = [
    {"mean": np.array([0, 0]), "cov": np.array([[0.5, 0], [0, 0.5]])},
    {"mean": np.array([5, 5]), "cov": np.array([[0.8, 0.3], [0.3, 0.8]])},
    {"mean": np.array([-5, 5]), "cov": np.array([[0.6, -0.2], [-0.2, 0.6]])},
    {"mean": np.array([5, -5]), "cov": np.array([[1.0, 0.5], [0.5, 1.0]])}
]

# Target log-probability density function
def target_log_density(position):
    return np.log(sum(multivariate_normal(mean=mode["mean"], cov=mode["cov"]).pdf(position) for mode in modes))

# Gradient of the log-probability density function
def target_log_density_gradient(position):
    grad = np.zeros(2)
    total_density = sum(multivariate_normal(mean=mode["mean"], cov=mode["cov"]).pdf(position) for mode in modes)
    
    for mode in modes:
        mean = mode["mean"]
        cov_inv = np.linalg.inv(mode["cov"])
        density = multivariate_normal(mean=mean, cov=mode["cov"]).pdf(position)
        grad += (density / total_density) * np.dot(cov_inv, (position - mean))
    
    return grad

# Reflective boundary condition to keep the sample within the domain
def reflect_position(position, domain_min, domain_max):
    position = np.clip(position, domain_min, domain_max)
    return position

# Initialize multiple chains with random starting positions spread across the domain
def initialize_chains(num_chains, domain_min, domain_max):
    positions = []
    for _ in range(num_chains):
        # Randomly initialize within the domain boundaries
        position = np.random.uniform(domain_min, domain_max, size=2)
        positions.append(position)
    return np.array(positions)

# Metropolis-Adjusted Langevin Algorithm (MALA) for multiple chains
def run_mala_chains(num_steps, burn_in_steps, step_size, num_chains, domain_min, domain_max):
    chains = initialize_chains(num_chains, domain_min, domain_max)
    all_samples = []

    for step in range(num_steps):
        for chain_idx in range(num_chains):
            position = chains[chain_idx]
            
            # Step 1: Calculate gradient of the log-density at the current position
            grad_log_density = target_log_density_gradient(position)
            
            # Step 2: Propose a new position using the MALA update rule
            proposal = position + (step_size / 2) * grad_log_density + np.sqrt(step_size) * np.random.normal(0, 1, 2)
            
            # Step 3: Reflect the proposal back into the domain if it's outside
            proposal = reflect_position(proposal, domain_min, domain_max)
            
            # Step 4: Calculate acceptance probability
            current_log_density = target_log_density(position)
            proposal_log_density = target_log_density(proposal)
            
            # Calculate reverse and forward proposal densities
            reverse_grad_log_density = target_log_density_gradient(proposal)
            forward_log_prob = -np.sum((proposal - (position + (step_size / 2) * grad_log_density)) ** 2) / (2 * step_size)
            reverse_log_prob = -np.sum((position - (proposal + (step_size / 2) * reverse_grad_log_density)) ** 2) / (2 * step_size)
            
            # Compute Metropolis acceptance probability
            acceptance_prob = np.exp(proposal_log_density + reverse_log_prob - current_log_density - forward_log_prob)
            acceptance_prob = min(1, acceptance_prob)

            # Accept or reject the proposal
            if np.random.rand() < acceptance_prob:
                chains[chain_idx] = proposal  # Accept the proposal and update the chain

        # Store the samples after burn-in period
        if step >= burn_in_steps:
            all_samples.extend(chains)  # Store samples from all chains
    
    return np.array(all_samples)

# Run MALA with multiple chains
samples = run_mala_chains(num_steps, burn_in_steps, step_size, num_chains, domain_min, domain_max)

# Create a grid for the background multimodal distribution
x = np.linspace(domain_min, domain_max, 300)
y = np.linspace(domain_min, domain_max, 300)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate the probability density for each mode and sum them up
Z = np.zeros(X.shape)
for mode in modes:
    rv = multivariate_normal(mean=mode["mean"], cov=mode["cov"])
    Z += rv.pdf(pos)

# Plotting
plt.figure(figsize=(10, 10))
# Plot the multimodal distribution as a contour or heatmap
plt.contourf(X, Y, Z, levels=50, cmap='Blues', alpha=0.5)

# Plot the MALA samples
plt.scatter(samples[:, 0], samples[:, 1], c=np.arange(len(samples)), cmap='viridis', s=5)
plt.colorbar(label="Sample Index")
plt.title("Samples from Multimodal Gaussian Distribution Using MALA with Multiple Chains")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.xlim(domain_min, domain_max)
plt.ylim(domain_min, domain_max)
plt.grid()
plt.show()
