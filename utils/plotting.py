import torch

import numpy as np
import matplotlib.pyplot as plt

from utils.transforms import complex_abs
from utils.temp_helper import prep_input_2_chan

def generate_image(fig, target, image, title, image_ind):
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(2, 3, image_ind)
    ax.set_title(title)
    ax.imshow(np.abs(image), cmap='gray', vmin=0, vmax=np.max(target))
    ax.set_xticks([])
    ax.set_yticks([])


def generate_error_map(fig, target, recon, image_ind, k=3, max=1):
    # Assume rows and cols are available globally
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(2, 3, image_ind)  # Add to subplot

    # Normalize error between target and reconstruction
    error = np.abs(target - recon)
    # normalized_error = error / error.max() if not relative else error
    im = ax.imshow(k * error, cmap='jet', vmax=max)  # Plot image

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Return plotted image and its axis in the subplot
    return im, ax
