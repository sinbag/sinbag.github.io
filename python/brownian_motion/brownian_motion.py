import numpy as np
import matplotlib.pyplot as plt

# Function to simulate Brownian motion
def brownian_motion(num_steps, step_size):
    # Initialize position at the origin
    position = np.zeros(2)
    positions = [position.copy()]

    # Iterate through each step
    for _ in range(num_steps):
        # Generate random displacement for each dimension
        displacement = np.random.normal(0, step_size, size=2)
        # Update position
        position += displacement
        # Store the updated position
        positions.append(position.copy())

    return np.array(positions)

# Number of steps and step size
num_steps = 1000
step_size = 0.1

# Simulate Brownian motion
positions = brownian_motion(num_steps, step_size)

# Plot the Brownian motion trajectory
plt.plot(positions[:, 0], positions[:, 1], lw=1)
plt.title('2D Brownian Motion')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
