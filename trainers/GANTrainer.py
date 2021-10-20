"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Adapted by: Matt Bendel based on work originally by Saurav

SPECIFICATIONS FOR TRAINING:
- Brain data onlu.
  - Can select contrast.
- GRO sampling pattern with R=4.
- All multicoil data where num_coils > 8 was condensed to 8 coil
- Can either train k-space U-Net or image space U-Net
- Base U-Net in either case has 16 input channels:
  - 8 per coil for real values
  - 8 per coil for complex values
"""
import shutil
import torch
import pytorch_ssim

import numpy as np
import torch.autograd as autograd

from typing import Optional
from utils.transforms import complex_abs
from utils.prepare_data import create_data_loaders
from utils.prepare_model import resume_train, fresh_start
from utils.temp_helper import prep_input_2_chan
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=maxval)

    return psnr_val


def ssim(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    # if not gt.ndim == 3:
    #   raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = structural_similarity(
        gt, pred, data_range=maxval
    )

    return ssim


def ssim_tensor(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    ssim_loss = pytorch_ssim.SSIM()
    return ssim_loss(gt, pred)


class GANTrainer:
    def __init__(self, args):
        args.exp_dir.mkdir(parents=True, exist_ok=True)

        self.args = args
        self.GLOBAL_LOSS_DICT = {
            'g_loss': [],
            'd_loss': [],
            'mSSIM': [],
            'd_acc': []
        }
        self.lambda_gp = 10

        if args.resume:
            self.generator, self.optimizer_G, self.discriminator, self.optimizer_D, self.args, self.best_dev_loss, self.start_epoch = resume_train(
                args)
        else:
            self.generator, self.discriminator, self.best_dev_loss, self.start_epoch = fresh_start(args)
            self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=args.lr,
                                                betas=(args.beta_1, args.beta_2))
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr,
                                                betas=(args.beta_1, args.beta_2))

        self.train_loader, self.dev_loader = create_data_loaders(args)

        if self.args.pretrained:
            temp = None

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        Tensor = torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.args.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.args.device)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def save_model(self, args, epoch, model, optimizer, best_dev_loss, is_new_best, m_type):
        torch.save(
            {
                'epoch': epoch,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dev_loss': best_dev_loss,
                'exp_dir': args.exp_dir
            },
            f=args.exp_dir / args.network_input / str(args.z_location) / f'{m_type}_model.pt'
        )

        if is_new_best:
            shutil.copyfile(args.exp_dir / args.network_input / str(args.z_location) / f'{m_type}_model.pt',
                            args.exp_dir / args.network_input / str(args.z_location) / f'{m_type}_best_model.pt'
                            )

    def get_zero_z(self, batch_len):
        if self.args.z_location == 1:
            z = torch.zeros((batch_len, self.args.latent_size))
        elif self.args.z_location == 2:
            # TODO: NEEDS CHANGED AT HIGHER RESOLUTION
            z = torch.zeros((batch_len, 512, 3, 3))
        else:
            z = None

        return z

    def get_z(self, batch_len):
        if self.args.z_location == 1:
            z = (0.001 ** 0.5) * torch.randn((batch_len, self.args.latent_size)).to(self.args.device)
        elif self.args.z_location == 2:
            # TODO: NEEDS CHANGED AT HIGHER RESOLUTION
            z = (0.001 ** 0.5) * torch.randn((batch_len, 512, 3, 3)).to(self.args.device)
        else:
            z = None

        return z

    def train_gen(self, input, target):
        self.optimizer_G.zero_grad()

        # mean = self.generator(input, self.get_zero_z(input.shape[0]))

        recon_list = []
        # variance_tensor = torch.zeros(input.shape)
        mean_tensor = torch.zeros(input.shape)
        for i in range(self.args.num_recons):
            z = self.get_z(input.shape[0])
            gen_val = self.generator(input, z)
            mean_tensor = torch.add(mean_tensor, gen_val)
            # variance_tensor = variance_tensor + (gen_val - mean) ** 2
            recon_list.append(self.generator(input, z))

        mean_tensor = torch.div(mean_tensor, self.args.num_recons)
        # variance_tensor = variance_tensor / self.args.num_recons

        # variance_2c_batch = torch.mean(variance_tensor, dim=(2, 3))
        # variance_gt = 0.25 * torch.ones(variance_2c_batch.shape)

        inds = np.random.choice(self.args.num_recons, 4)
        disc_output_1 = self.discriminator(recon_list[inds[0]])
        disc_output_2 = self.discriminator(recon_list[inds[1]])
        disc_output_3 = self.discriminator(recon_list[inds[2]])
        disc_output_4 = self.discriminator(recon_list[inds[3]])
        disc_output_cat = torch.cat((disc_output_1, disc_output_2, disc_output_3, disc_output_4))

        adversarial_loss = -torch.mean(disc_output_cat)
        mean_loss = -10 * ssim_tensor(target, mean_tensor)
        # variance_loss = F.l1_loss(variance_2c_batch, variance_gt)
        g_loss = adversarial_loss + mean_loss

        g_loss.backward()
        self.optimizer_G.step()

        return g_loss.item()

    def train_dis(self, input, target):
        for j in range(self.args.num_iters_discriminator):
            z = self.get_z(input.shape[0])

            self.optimizer_D.zero_grad()

            out_gen = self.generator(input, z)

            print(out_gen.device)
            print(target.device)
            print('\n')

            # MAKE PREDICTIONS
            real_pred = self.discriminator(target)
            fake_pred = self.discriminator(out_gen)

            real_acc = real_pred[real_pred > 0].shape[0]
            fake_acc = fake_pred[fake_pred <= 0].shape[0]

            d_acc = (real_acc + fake_acc) / 32

            # Gradient penalty
            gradient_penalty =self. compute_gradient_penalty(self.discriminator, target.data, out_gen.data)
            # Adversarial loss
            d_loss = torch.mean(fake_pred) - torch.mean(real_pred) + self.lambda_gp * gradient_penalty

            d_loss.backward()
            self.optimizer_D.step()

        return d_loss.item(), d_acc

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()

        batch_loss = {
            'g_loss': [],
            'd_loss': [],
            'd_acc': []
        }

        for i, data in enumerate(self.train_loader):
            input, target, mean, std, nnz_index_mask = data
            input = input.to(self.args.device)
            target = target.to(self.args.device)

            prepped = prep_input_2_chan(input, self.args.device)
            target = prep_input_2_chan(target, self.args.device)

            d_loss, d_acc = self.train_dis(prepped, target)
            g_loss = self.train_gen(prepped, target)

            batch_loss['d_acc'].append(d_acc)
            batch_loss['g_loss'].append(g_loss.item())
            batch_loss['d_loss'].append(d_loss.item())

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]"
                % (epoch + 1, self.args.num_epochs, i, len(self.train_loader.dataset) / self.args.batch_size, d_loss,
                   g_loss)
            )

        self.GLOBAL_LOSS_DICT['g_loss'].append(np.mean(batch_loss['g_loss']))
        self.GLOBAL_LOSS_DICT['d_loss'].append(np.mean(batch_loss['d_loss']))
        self.GLOBAL_LOSS_DICT['d_acc'].append(np.mean(batch_loss['d_acc']))

    def validate_epoch(self):
        self.generator.eval()
        self.discriminator.eval()

        mSSIM = []
        mPSNR = []

        with torch.no_grad():
            for i, data in enumerate(self.dev_loader):
                input, target, mean, std, nnz_index_mask = data

                input = input.to(self.args.device)
                target = target.to(self.args.device)

                prepped = prep_input_2_chan(input, self.args.device)
                target = prep_input_2_chan(target, self.args.device)

                gen_out = self.generator(prepped, self.get_z(prepped.shape[0]))

                for j in range(gen_out.shape[0]):
                    gen_im = complex_abs(gen_out[j].permute(1, 2, 0))
                    true_im = complex_abs(target[j].permute(1, 2, 0))

                    gen_im_np = gen_im.cpu().numpy() * std[j].numpy() + mean[j].numpy()
                    true_im_np = true_im.cpu().numpy() * std[j].numpy() + mean[j].numpy()

                    mSSIM.append(ssim(true_im_np, gen_im_np))
                    mPSNR.append(psnr(true_im_np, gen_im_np))

                if i == 10:
                    break

        return np.mean(mSSIM), np.mean(mPSNR)

    def train(self):
        best_loss = 0
        for epoch in range(self.args.num_epochs):
            batch_loss = {
                'g_loss': [],
                'd_loss': [],
                'd_acc': []
            }
            self.train_epoch(epoch)

            ssim_val, psnr = self.validate_epoch()

            best_model = ssim_val > best_loss  # val_data()
            best_loss_val = ssim_val if best_model else best_loss
            best_loss = best_loss_val

            save_str = f"END OF EPOCH {epoch + 1}: " \
                       f"[Average D loss: {self.GLOBAL_LOSS_DICT['d_loss'][epoch]:.4f}] " \
                       f"[Average D Acc: {self.GLOBAL_LOSS_DICT['d_acc'][epoch]:.4f}] " \
                       f"[Average G loss: {self.GLOBAL_LOSS_DICT['g_loss'][epoch]:.4f}]" \
                       f"[Val SSIM: {ssim_val:.4f}]" \
                       f"[Val PSNR: {psnr:.2f}]\n"

            print(save_str)

            self.save_model(self.args, epoch, self.generator, self.optimizer_G, best_loss_val, best_model, 'generator')
            self.save_model(self.args, epoch, self.discriminator, self.optimizer_D, best_loss_val, best_model,
                            'discriminator')
