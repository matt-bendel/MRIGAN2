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


def generate_error_map(fig, target, recon, image_ind, k=5, max=1):
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


def plot_epoch(args, generator, epoch, CONSTANT_PLOTS):
    std = CONSTANT_PLOTS['std']
    mean = CONSTANT_PLOTS['mean']

    z_1 = CONSTANT_PLOTS['measures'].unsqueeze(0).to(args.device)
    z = torch.FloatTensor(np.random.normal(size=(z_1.shape[0], args.latent_size))).to(args.device)

    generator.eval()
    with torch.no_grad():
        z_1_out = generator(input=z_1, z=z)

    target_prep = CONSTANT_PLOTS['gt']
    zfr = CONSTANT_PLOTS['measures']
    z_1_prep = z_1_out[0]

    target_im = complex_abs(target_prep.permute(1, 2, 0)) * std + mean
    target_im = target_im.numpy()

    zfr = complex_abs(zfr.permute(1, 2, 0)) * std + mean
    zfr = zfr.numpy()

    z_1_im = complex_abs(z_1_prep.permute(1, 2, 0)) * std + mean
    z_1_im = z_1_im.detach().cpu().numpy()

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(f'Generated and GT Images at Epoch {epoch + 1}')
    generate_image(fig, target_im, target_im, 'GT', 1)
    generate_image(fig, target_im, zfr, 'ZFR', 2)
    generate_image(fig, target_im, z_1_im, 'Z 1', 3)

    max_val = np.max(np.abs(target_im - zfr))
    generate_error_map(fig, target_im, zfr, 5, 1, max_val)
    generate_error_map(fig, target_im, z_1_im, 6, 1, max_val)

    plt.savefig(
        f'/home/bendel.8/Git_Repos/MRIGAN2/training_images/gen_{args.z_location}_{epoch + 1}.png')