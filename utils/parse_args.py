import pathlib
import argparse


class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
        self.add_argument('--resolution', default=320, type=int, help='Resolution of images')

        # Data parameters
        self.add_argument('--challenge', choices=['singlecoil', 'multicoil'], default='singlecoil',
                          help='Which challenge')
        self.add_argument('--data-path', type=pathlib.Path, required=True,
                          help='Path to the dataset')
        self.add_argument('--sample-rate', type=float, default=1.,
                          help='Fraction of total volumes to include')

        # Mask parameters
        self.add_argument('--accelerations', nargs='+', default=[4], type=int,
                          help='Ratio of k-space columns to be sampled. If multiple values are '
                               'provided, then one of those is chosen uniformly at random for '
                               'each volume.')
        self.add_argument('--center-fractions', nargs='+', default=[0.08], type=float,
                          help='Fraction of low-frequency k-space columns to be sampled. Should '
                               'have the same length as accelerations')

        # Override defaults with passed overrides
        self.set_defaults(**overrides)


def create_arg_parser():
    # CREATE THE PARSER
    parser = Args()

    # GAN ARGS
    parser.add_argument('--num-iters-discriminator', type=int, default=5,
                        help='Number of iterations of the discriminator')
    parser.add_argument('--latent-size', type=int, default=512, help='Size of latent vector for z location 2')
    parser.add_argument('--z-location', type=int, required=True, help='Where to put code vector')
    parser.add_argument('--pretrained', action='store_true', help='Whether or not to freeze left half weights')
    parser.add_argument('--num_recons', type=int, default=4, help='Number of recons')
    parser.add_argument('--in-chans', type=int, default=2, help='Number of input channels')
    parser.add_argument('--out-chans', type=int, default=2, help='Number of output channels')

    # LEARNING ARGS
    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=10e-4, help='Learning rate')
    parser.add_argument('--beta_1', type=float, default=0, help='Beta 1 for Adam')
    parser.add_argument('--beta_2', type=float, default=0.9, help='Beta 2 for Adam')

    # DATA ARGS
    parser.add_argument('--data-parallel', required=True, action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--num_of_top_slices', default=6, type=int,
                        help='top slices have bigger brain image and less air region')
    parser.add_argument('--use-middle-slices', action='store_true',
                        help='If set, only uses central slice of every data collection')

    # LOGISTICAL ARGS
    parser.add_argument('--device', type=int, default=0,
                        help='Which device to train on. Use idx of cuda device or -1 for CPU')
    parser.add_argument('--exp-dir', type=pathlib.Path,
                        default=pathlib.Path('/home/bendel.8/Git_Repos/MRIGAN/trained_models'),
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint_gen', type=str,
                        help='Path to an existing generator checkpoint. Used along with "--resume"')
    parser.add_argument('--checkpoint_dis', type=str,
                        help='Path to an existing discriminator checkpoint. Used along with "--resume"')
    parser.add_argument('--checkpoint_unet', type=str,
                        help='Path to an existing unet checkpoint. Used along with "--resume"')

    return parser
