import torch
import os
import random
import pathlib

import numpy as np

from trainers.GANTrainer import GANTrainer
from utils.parse_args import create_arg_parser

if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.FloatTensor

    args = create_arg_parser().parse_args()
    args.exp_dir = pathlib.Path('/home/bendel.8/Git_Repos/MRIGAN2/trained_models')
    # restrict visible cuda devices
    if args.data_parallel or (args.device >= 0):
        if not args.data_parallel:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        args.device = torch.device('cuda')
        print(args.device)
    else:
        args.device = torch.device('cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainer = GANTrainer(args)
    trainer.train()