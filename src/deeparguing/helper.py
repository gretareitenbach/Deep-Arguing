from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import *


def exclude_fields(X, excluded_indices):
    all_indices = np.arange(X.shape[-1])
    mask = np.logical_not(np.isin(all_indices, excluded_indices))
    return X[:, mask]


def one_hot_encode(data):
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(data)
    return encoder.transform(data)


def load_tabular_data(
    path: str,
    target_field: str,
    device: str = "cpu",
    labels: list[str] = [],
    size: int | None = None,
    shuffle: bool = False,
    seed: float = 42,
    excluded_fields: list[str] = [],
    one_hot_X: bool = False,
    has_header: bool = False,
):
    if has_header:
        data = pd.read_csv(path, header=0)
    else:
        data = pd.read_csv(path, header=None)

    data = data.values
    X = exclude_fields(np.array(data), [target_field] + excluded_fields)
    y = data[:, target_field]

    if shuffle:
        np.random.seed(seed)
        indicies = np.random.permutation(len(X))
        X = X[indicies]
        y = y[indicies]

    if one_hot_X:
        X = one_hot_encode(X)

    if len(labels) != 0:
        mask = np.isin(y, labels)
        y = y[mask]
        X = X[mask]

    y = y.reshape(-1, 1)
    y = one_hot_encode(y)

    X = np.array(X, dtype=np.float32)
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    X = X[:size]
    y = y[:size]

    return X, y


def load_torch_images(
    class_name: str,
    device: str = "cpu",
    as_vector: bool = False,
    size: int = -1,
    labels: list[int] = [],
    shuffle: bool = False,
    seed: float = 42,
) -> Tuple[Tensor, Tensor]:
    """
    Loads an image dataset (e.g., MNIST, CIFAR10) directly as torch tensors.

    Args:
        class_name: The name of a torchvision dataset class (string name, e.g. 'MNIST').
        device: Device for tensors ('cpu' or 'cuda').
        as_vector: Flatten images into vectors if True.
        size: Limit the dataset to this many samples (-1 for all).
        labels: Optional list of label integers to filter by.
        shuffle: Shuffle the data if True.
        seed: Random seed for reproducibility.

    Returns:
        (X, y): Tensors of images and one-hot encoded labels.
    """

    dataset_class = globals()[class_name]
    dataset = dataset_class("./temp/", train=True, download=True)

    if size == -1 or size > len(dataset):
        size = len(dataset)

    X = dataset.data[:size]
    y = torch.tensor(dataset.targets[:size], dtype=torch.long)

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    X = X.float()

    if len(labels) > 0:
        mask = torch.isin(y, torch.tensor(labels, device=y.device))
        X = X[mask]
        y = y[mask]

    X = X.permute(0, 3, 1, 2).float() / 255.0
    X = (
        X - torch.tensor([0.4914, 0.4822, 0.4465])[None, :, None, None]
    ) / torch.tensor([0.2470, 0.2435, 0.2616])[None, :, None, None]

    if as_vector:
        X = X.view(X.size(0), -1)

    if shuffle:
        torch.manual_seed(int(seed))
        indices = torch.randperm(X.size(0))
        X = X[indices]
        y = y[indices]

    y = torch.nn.functional.one_hot(y, num_classes=len(torch.unique(y))).float()

    # Move to device
    X = X.to(device)
    y = y.to(device)

    return X, y


def split_data(
    X: Tensor, y: Tensor, seed: float, test_size: float = 0.2
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=test_size, random_state=seed
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(X: Tensor, mean: float, std: float) -> Tensor:
    return (X - mean) / std
