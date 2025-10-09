import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.animation import FuncAnimation

# Parameters
# rng = np.random.default_rng(2021)
# rng = np.random.RandomState(2021)
np.random.seed(12328)
num_steps = 4000
jump_prob = 0.05  # Probability of jumping to a different Gaussian distribution
proposal_std_dev = 0.5  # Standard deviation for the proposal distribution

# Define the centers and covariances of each Gaussian mode
modes = [
    {"mean": np.array([0, 0]), "cov": np.array([[0.5, 0], [0, 0.5]])},
    {"mean": np.array([5, 5]), "cov": np.array([[0.8, 0.3], [0.3, 0.8]])},
    {"mean": np.array([-5, 5]), "cov": np.array([[0.6, -0.2], [-0.2, 0.6]])},
    {"mean": np.array([5, -5]), "cov": np.array([[1.0, 0.5], [0.5, 1.0]])}
]

# Compute the target density at a given position by summing densities of all modes
def target_density(position):
    return sum(multivariate_normal(mean=mode["mean"], cov=mode["cov"]).pdf(position) for mode in modes)


# Initial position
location = np.array([0.0, 0.0])

# Store the path
positions = [location.copy()]

# Function to choose a random mode with probability
def choose_mode():
    return np.random.choice(len(modes))


def brownian_motion(location):
    # Simulate the Brownian motion with multimodal Gaussian jumps
    for _ in range(num_steps):
        # Regular Brownian motion step (small random walk)
        step = np.random.normal(0, 0.1, 2)
        location += step

        # Append the current position to the path
        positions.append(location.copy())
    # return positions



def brownian_motion_wjumps(location):
    # Simulate the Brownian motion with multimodal Gaussian jumps
    for _ in range(num_steps):
        if np.random.rand() < jump_prob:
            # Jump to a new mode
            mode = choose_mode()
            mean = modes[mode]["mean"]
            cov = modes[mode]["cov"]
            
            # New position based on the chosen Gaussian mode
            location = np.random.multivariate_normal(mean, cov)
        else:
            # Regular Brownian motion step (small random walk)
            step = np.random.normal(0, 0.1, 2)
            location += step

        # Append the current position to the path
        positions.append(location.copy())
    # return positions


# Metropolis-Hastings Algorithm for Brownian Motion (with no explicit jumps across modes)
def brownian_motion_metropolis_adjusted(location):
    for _ in range(num_steps):
        # Step 1: Propose a new position from a Gaussian centered at the current position
        proposal = location + np.random.normal(0, proposal_std_dev, 2)

        # Step 2: Calculate the acceptance probability
        current_density = target_density(location)
        proposal_density = target_density(proposal)
        acceptance_prob = min(1, proposal_density / current_density)

        # Step 3: Accept or reject the proposal
        if np.random.rand() < acceptance_prob:
            location = proposal  # Accept the proposal and move to the new position

        # Append the position to the path
        positions.append(location.copy())
    # return positions

# Metropolis-Hastings Algorithm for Brownian Motion (with random jumps across modes)
def brownian_motion_metropolis_adjusted_wjumps(location):
    for _ in range(num_steps):
        if np.random.rand() < jump_prob:
            # Jump to a new mode
            mode = choose_mode()
            mean = modes[mode]["mean"]
            cov = modes[mode]["cov"]
            
            # New position based on the chosen Gaussian mode
            location = np.random.multivariate_normal(mean, cov)
        else:
            # Step 1: Propose a new position from a Gaussian centered at the current position
            proposal = location + np.random.normal(0, proposal_std_dev, 2)

            # Step 2: Calculate the acceptance probability
            current_density = target_density(location)
            proposal_density = target_density(proposal)
            acceptance_prob = min(1, proposal_density / current_density)

            # Step 3: Accept or reject the proposal
            if np.random.rand() < acceptance_prob:
                location = proposal  # Accept the proposal and move to the new position

            # Append the position to the path
            positions.append(location.copy())
    # return positions

# Convert path to numpy array for easy plotting
# brownian_motion(location)
# brownian_motion_wjumps(location)
# brownian_motion_metropolis_adjusted(location)
brownian_motion_metropolis_adjusted_wjumps(location)
positions = np.array(positions)

# Plotting
# plt.figure(figsize=(10, 10))

# Plot the result
fig, ax = plt.subplots()
ax.set_xlim([-8, 8])
ax.set_ylim([-8, 8])
ax.set_aspect('equal')
# Plot the resulting distribution
ax.scatter(positions[:, 0], positions[:, 1], color = 'green', s=10, alpha=1)
# ax.plot(positions[0, 0], positions[0, 1], 'ro', label='Start')


# Create a grid for the background multimodal distribution
x = np.linspace(-10, 10, 300)
y = np.linspace(-10, 10, 300)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate the probability density for each mode and sum them up
Z = np.zeros(X.shape)
for mode in modes:
    rv = multivariate_normal(mean=mode["mean"], cov=mode["cov"])
    Z += rv.pdf(pos)

# Plot the multimodal distribution as a contour or heatmap
ax.contourf(X, Y, Z, levels=50, cmap='Oranges', alpha=0.5)


# Hide axes, ticks, and labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Remove the legend
ax.legend().set_visible(False)


# To animate the brownian motion
line, = ax.plot([], [], 'b-', label='Brownian motion', alpha=0.2)
dot, = ax.plot([], [], 'go', markersize=2)
# dot, = ax.scatter(positions[:, 0], positions[:, 1], color = 'green', s=10)


# Initialization function
def init():
    line.set_data([], [])
    dot.set_data([], [])
    return line, dot


# Update function for animation
def update(frame):
    line.set_data(positions[:frame,0], positions[:frame,1])
    dot.set_data(positions[:frame,0], positions[:frame,1])
    return line, dot

# Create animation
ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=5)
# ani.save(filename="bmotion_multimodal.mp4", writer="ffmpeg")
# ani.save(filename="bmotion_multimodal_wjumps.mp4", writer="ffmpeg")
# ani.save(filename="bmotion_multimodal_metropolis_adjusted.mp4", writer="ffmpeg")
# ani.save(filename="bmotion_multimodal_metropolis_adjusted_with_jumps.mp4", writer="ffmpeg")

# plt.title("Brownian Motion with Disjoint Multimodal Gaussian Jumps")
# plt.savefig('target_multimodal_gaussian.pdf',bbox_inches='tight')
# plt.savefig('bmotion_multimodal.pdf',bbox_inches='tight')
# plt.savefig('bmotion_multimodal_wjumps.pdf',bbox_inches='tight')
plt.savefig('bmotion_multimodal_metropolis_adjusted_wjumps.pdf',bbox_inches='tight')
# plt.savefig('bmotion_multimodal_metropolis_adjusted.pdf',bbox_inches='tight')
plt.show()

# plt.plot(path[:, 0], path[:, 1], '-o', markersize=2, linewidth=0.5)
# plt.title("Brownian Motion with Disjoint Multimodal Gaussian Jumps")
# plt.xlabel("X position")
# plt.ylabel("Y position")
# plt.grid()
# plt.show()
