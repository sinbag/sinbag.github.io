import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from PIL import Image as im 
from matplotlib import cm

# Generate data for a bimodal Gaussian distribution
x = np.linspace(-10, 10, 256)


def visualize_diffusion_process():

    # mean1, mean2 = 0, 0
    mean1, mean2 = -3, 4
    std1, std2 = 2, 2

    array = np.arange(0, 256000, 1, np.uint8) 
    # reshape to desired shape
    array = np.reshape(array, (1000, 256)) 

    for i in range(100):

        # def gaussian_bimodal(x):
        gaussian1 = np.exp(-0.5 * ((x - mean1) / std1)**2) / (std1 * np.sqrt(2 * np.pi))
        gaussian2 = np.exp(-0.5 * ((x - mean2) / std2)**2) / (std2 * np.sqrt(2 * np.pi))
        bimodal = gaussian1 + gaussian2
        # return bimodal

        # Normalize the distribution for proper colormap application
        bimodal_normalized = bimodal / bimodal.max()

        for j in range(len(x) - 1):
            # draw each gaussian state as 10-pixel wide
            curr_ind = i*10
            new_ind = (i+1)*10
            array[curr_ind:new_ind,j] = bimodal_normalized[j]*255
        
        mean1 += (abs(mean1) * 0.01)
        mean2 -= (abs(mean2) * 0.01)

    # creating image object of above array 
    data = im.fromarray(array)

    # plt.imshow(data, cmap='viridis')
    plt.imsave('test.png', data)



def visualize_data_prior_image():
    mean1, mean2 = 0, 0
    std1, std2 = 2, 2

    # def gaussian_bimodal(x):
    gaussian1 = np.exp(-0.5 * ((x - mean1) / std1)**2) / (std1 * np.sqrt(2 * np.pi))
    gaussian2 = np.exp(-0.5 * ((x - mean2) / std2)**2) / (std2 * np.sqrt(2 * np.pi))
    bimodal = gaussian1 + gaussian2

    # Normalize the distribution for proper colormap application
    bimodal_normalized = bimodal / bimodal.max()

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the bimodal Gaussian distribution with a Viridis colormap
    for i in range(len(x) - 1):
        ax.fill_between(x[i:i+2], bimodal[i:i+2], color=viridis(bimodal_normalized[i]), linewidth=0)

    # Add labels, title, and grid
    # ax.set_title("1D Bimodal Gaussian with Viridis Colormap", fontsize=14)
    # ax.set_xlabel("X-axis", fontsize=12)
    # ax.set_ylabel("Density", fontsize=12)
    # ax.grid(alpha=0.3)
    # Hide axes, ticks, and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Save the figure as an image
    plt.savefig("bimodal_gaussian.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    visualize_data_prior_image()

# driver code 
if __name__ == "__main__": 
    
  # function call 
  main() 