from typing import Iterable

import numpy as np
import torch
from scipy.stats import norm
from tqdm import tqdm


def MLDL(
    label: Iterable[Iterable[int]], size: int, std: float = 1, scaling: bool = True
) -> torch.Tensor:
    """Maximum of Gaussians labeling over positions 0..size-1 for each sample."""
    # Convert to list to avoid consuming iterables twice
    label_list = list(label)
    z = np.zeros((len(label_list), size))
    for i, j in enumerate(tqdm(label_list)):
        for t in j:
            z[i] = np.maximum(z[i], np.array(norm.pdf(range(size), loc=t, scale=std)))
        if scaling and np.max(z[i]) > 0:
            z[i] = z[i] / np.max(z[i])
    z = torch.from_numpy(z).clone()
    return z


def one_hot(label: Iterable[int], size: int) -> torch.Tensor:
    """One-hot encode integer labels.

    Args:
        label: Iterable of integer class indices.
        size: Total number of classes.
    """
    label_list = list(label)
    onehot = torch.zeros(len(label_list), size)
    for i, j in enumerate(tqdm(label_list)):
        onehot[i, j] = 1
    return onehot
