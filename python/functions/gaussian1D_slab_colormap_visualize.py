import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from PIL import Image as im 

# Generate data for a bimodal Gaussian distribution
x = np.linspace(-10, 10, 256)
# mean1, mean2 = 0, 0
# mean1, mean2 = -3, 4
# mean1, mean2 = -2, 3
mean1, mean2 = -1.5, 2.5
std1, std2 = 1, 2

# def gaussian_bimodal(x):
gaussian1 = np.exp(-0.5 * ((x - mean1) / std1)**2) / (std1 * np.sqrt(2 * np.pi))
gaussian2 = np.exp(-0.5 * ((x - mean2) / std2)**2) / (std2 * np.sqrt(2 * np.pi))
bimodal = gaussian1 + gaussian2
# return bimodal

# Normalize the distribution for proper colormap application
bimodal_normalized = bimodal / bimodal.max()

array = np.arange(0, 2560, 1, np.uint8) 

# reshape to desired shape
array = np.reshape(array, (10, 256)) 
        
for i in range(len(x) - 1):
    array[:,i] = bimodal_normalized[i]*255

# creating image object of above array 
data = im.fromarray(array)

plt.imshow(data, cmap='viridis')
plt.imsave('test.png', data)

