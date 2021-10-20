"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import pickle
import random
import pathlib
import h5py
import torch
import yaml

import xml.etree.ElementTree as etree
import xml.etree.ElementTree as ET
import numpy as np

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn


def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def fetch_dir(
    key: str, data_config_file: Union[str, Path, os.PathLike] = "fastmri_dirs.yaml"
) -> Path:
    """
    Data directory fetcher.
    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.
    Args:
        key: key to retrieve path from data_config_file. Expected to be in
            ("knee_path", "brain_path", "log_path").
        data_config_file: Optional; Default path config file to fetch path
            from.
    Returns:
        The path to the specified directory.
    """
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        default_config = {
            "knee_path": "/path/to/knee",
            "brain_path": "/path/to/brain",
            "log_path": ".",
        }
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        data_dir = default_config[key]

        warn(
            f"Path config at {data_config_file.resolve()} does not exist. "
            "A template has been created for you. "
            "Please enter the directory paths for your system to have defaults."
        )
    else:
        with open(data_config_file, "r") as f:
            data_dir = yaml.safe_load(f)[key]

    return Path(data_dir)


class SelectiveSliceData(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1, use_top_slices=True, number_of_top_slices=6, restrict_size=False):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' else 'reconstruction_rss'

        self.examples = []
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        files = list(pathlib.Path(root).iterdir())

        # remove files with wrong modality or scanner
        keep_files = []
        f = sorted(files)

        for fname in f[1:]:
            with h5py.File(fname, 'r') as data:
                if (data.attrs['acquisition'] == 'AXT2'):
                    keep_files.append(fname)

        files = keep_files

        random.seed(1000)
        np.random.seed(1000)

        random.shuffle(files)

        num_files = round(len(files) * 0.7)

        f_training = sorted(files[0:num_files])

        files = f_training

        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]

        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            if kspace.shape[-1] <= 384 or kspace.shape[1] < 10 or str(
                    fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_209_2090296.h5' or str(
                    fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_200_2000250.h5' or str(
                    fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_201_2010106.h5' or str(
                    fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_204_2130024.h5' or str(
                    fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_210_2100025.h5':
                continue
            else:
                if restrict_size and ((kspace.shape[1] != 640) or (kspace.shape[2] != 368)):
                    continue  # skip non uniform sized images
                num_slices = kspace.shape[0]
                if use_top_slices:
                    start_idx = 0
                    end_idx = start_idx + number_of_top_slices
                    self.examples += [(fname, slice) for slice in range(start_idx, end_idx)]
                else:
                    self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace, target, data.attrs, fname.name, slice)


class SelectiveSliceData_Val(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1, use_top_slices=True, number_of_top_slices=6, restrict_size=False):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' else 'reconstruction_rss'

        self.examples = []
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        files = list(pathlib.Path(root).iterdir())

        # remove files with wrong modality or scanner
        keep_files = []
        f = sorted(files)

        for fname in f[1:len(f)]:
            kspace = h5py.File(fname, 'r')['kspace']
            with h5py.File(fname, 'r') as data:
                if (data.attrs['acquisition'] == 'AXT2'):
                    if kspace.shape[1] >= 8:
                        keep_files.append(fname)

        files = keep_files

        print(len(files))

        random.seed(1000)
        np.random.seed(1000)

        random.shuffle(files)

        num_files = round(len(files)*0.3)

        f_testing_and_Val = sorted(files[0:num_files])

        files = f_testing_and_Val

        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]

        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']

            if kspace.shape[-1] <= 384 or kspace.shape[1] < 10 or str(
                    fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_209_2090296.h5' or str(
                    fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_200_2000250.h5' or str(
                    fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_201_2010106.h5' or str(
                    fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_204_2130024.h5' or str(
                    fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_210_2100025.h5':
                continue
            else:
                if restrict_size and ((kspace.shape[1] != 640) or (kspace.shape[2] != 368)):
                    continue  # skip non uniform sized images
                num_slices = kspace.shape[0]
                if use_top_slices:
                    start_idx = 0
                    end_idx = start_idx + number_of_top_slices
                    self.examples += [(fname, slice) for slice in range(start_idx, end_idx)]
                else:
                    self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace, target, data.attrs, fname.name, slice)