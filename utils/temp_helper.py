import torch
import cv2

import numpy as np

from utils import transforms

def readd_measures_im(data_tensor, old, args):
    im_size = 96
    disc_inp = torch.zeros(data_tensor.shape[0], 2, im_size, im_size).to(args.device)

    for k in range(data_tensor.shape[0]):
        output = torch.squeeze(data_tensor[k])
        output_tensor = transforms.fft2c(output.permute(1, 2, 0))

        old_out = torch.squeeze(old[k])
        old_out = transforms.fft2c(old_out.permute(1, 2, 0))

        disc_inp[k, :, :, :] = output_tensor.permute(2, 0, 1) + old_out.permute(2, 0, 1)

    for k in range(data_tensor.shape[0]):
        output = torch.squeeze(disc_inp[k])
        output_tensor = transforms.ifft2c(output.permute(1, 2, 0))

        disc_inp[k, :, :, :] = output_tensor.permute(2, 0, 1)

    return disc_inp


def prep_input_2_chan(data_tensor):
    im_size = 96
    disc_inp = torch.zeros(data_tensor.shape[0], 2, im_size, im_size)

    for k in range(data_tensor.shape[0]):
        output = torch.squeeze(data_tensor[k])
        output_tensor = torch.zeros(8, 384, 384, 2)
        output_tensor[:, :, :, 0] = output[0:8, :, :]
        output_tensor[:, :, :, 1] = output[8:16, :, :]
        output_x = transforms.ifft2c(output_tensor)
        output_x = transforms.rss(output_x)
        # REMOVE BELOW TWO LINES TO GO BACK UP
        output_x_r = cv2.resize(output_x[:, :, 0].numpy(), dsize=(96, 96), interpolation=cv2.INTER_LINEAR)
        output_x_c = cv2.resize(output_x[:, :, 1].numpy(), dsize=(96, 96), interpolation=cv2.INTER_LINEAR)

        output_x_r = torch.from_numpy(output_x_r).unsqueeze(-1)
        output_x_c = torch.from_numpy(output_x_c).unsqueeze(-1)
        ######################################
        output_x = torch.cat((output_x_r, output_x_c), dim=-1)

        disc_inp[k, :, :, :] = output_x.permute(2, 0, 1)

    return disc_inp
