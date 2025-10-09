import numpy as np
import matplotlib.pyplot as plt

# Parameters of the Gaussian
mu = 0      # Mean
sigma = 1   # Standard deviation

# Define the Gaussian probability density function (PDF)
def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Define the score function of the Gaussian
def score_function(x, mu, sigma):
    return -(x - mu) / sigma**2

# Generate data points
x = np.linspace(-3 * sigma, 3 * sigma, 500)

# Compute the Gaussian PDF and score function
pdf = gaussian_pdf(x, mu, sigma)
score = score_function(x, mu, sigma)

# Plotting
plt.figure(figsize=(10, 6))

# Plot the Gaussian PDF
plt.plot(x, pdf, label="Gaussian PDF", color='blue', linewidth=2)

# Plot the score function
plt.plot(x, score, label="Score Function", color='red', linestyle='--', linewidth=2)

# Add labels and legend
plt.title("Gaussian PDF and its Score Function", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.legend(fontsize=12)
plt.grid(alpha=0.5)

# Show the plot
plt.show()
