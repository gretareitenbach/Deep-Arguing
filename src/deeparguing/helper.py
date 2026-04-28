from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import *


def exclude_fields(X, excluded_indices):
    all_indices = np.arange(X.shape[-1])
    mask = np.logical_not(np.isin(all_indices, excluded_indices))
    return X[:, mask], mask


def apply_binary_map(X, column_indices, binary_maps):
    """
    Apply binary mappings column-wise
    """
    for col in column_indices:
        mapping = binary_maps[str(col)]
        X[:, col] = np.vectorize(lambda x: mapping.get(x, 0))(X[:, col])
    return X


def apply_scaling(X, column_indices):
    scaler = StandardScaler()
    X[:, column_indices] = scaler.fit_transform(X[:, column_indices].astype(float))
    return X


def apply_one_hot(X, column_indices):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    subset = X[:, column_indices]

    encoded = encoder.fit_transform(subset)

    remaining_indices = [i for i in range(X.shape[1]) if i not in column_indices]

    remaining = X[:, remaining_indices]

    return np.concatenate([remaining, encoded], axis=1)


def one_hot_encode(data):
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(data)

    categories = encoder.categories_[0]
    print("Label One-Hot Encoding Mapping:")
    for i, category in enumerate(categories):
        one_hot_vec = [0] * len(categories)
        one_hot_vec[i] = 1
        print(f"  {category} -> {one_hot_vec} -> Class {i}")

    return encoder.transform(data)


def load_tabular_data(
    path: str,
    target_field: int,
    device: str = "cpu",
    labels: List[str] = [],
    size: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
    excluded_fields: List[int] = [],
    has_header: bool = False,
    categorical_cols: List[int] = [],
    binary_cols: List[int] = [],
    continuous_cols: List[int] = [],
    binary_maps: Optional[Dict[int, Dict]] = None,
    sep: str = ",",
):
    """
    Generic tabular dataset loader with flexible preprocessing support.
    Column selections must be passed as integer indices.
    """

    # ---------- load data ----------

    if has_header:
        data = pd.read_csv(path, header=0, sep=sep)
    else:
        data = pd.read_csv(path, header=None, sep=sep)

    data = data.values

    # ---------- split X / y ----------

    X = exclude_fields(np.array(data), [target_field] + excluded_fields)[0]

    y = data[:, target_field]

    # ---------- label filtering ----------

    if len(labels) != 0:
        mask = np.isin(y, labels)
        y = y[mask]
        X = X[mask]

    # ---------- shuffle ----------

    if shuffle:
        np.random.seed(seed)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

    # ---------- binary encoding ----------

    if binary_cols:

        if binary_maps is None:
            raise ValueError("binary_maps must be provided if binary_cols specified")

        X = apply_binary_map(X, binary_cols, binary_maps)

    # ---------- scaling continuous columns ----------

    if continuous_cols:
        X = apply_scaling(X, continuous_cols)

    # ---------- one-hot encode categorical columns ----------

    if categorical_cols:
        X = apply_one_hot(X, categorical_cols)

    # ---------- encode target ----------

    y = y.reshape(-1, 1)
    y = one_hot_encode(y)

    # ---------- truncate dataset ----------

    if size is not None:
        X = X[:size]
        y = y[:size]

    # ---------- convert to tensors ----------

    X = np.array(X, dtype=np.float32)

    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    return X, y


def load_torch_images(
    class_name: str,
    device: str = "cpu",
    as_vector: bool = False,
    size: int = -1,
    labels: list[int] = [],
    shuffle: bool = False,
    seed: float = 42,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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

    X = dataset.data
    y = torch.tensor(dataset.targets, dtype=torch.long)

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    X = X.float()

    if len(labels) > 0:
        mask = torch.isin(y, torch.tensor(labels, device=y.device))
        X = X[mask]
        y = y[mask]

    X = X[:size]
    y = y[:size]

    if X.ndim == 4:
        X = X.permute(0, 3, 1, 2).float() / 255.0
    else:
        X = X.unsqueeze(1).float() / 255.0


    mean = X.mean(dim=(0, 2, 3), keepdim=True)
    std = X.std(dim=(0, 2, 3), keepdim=True)
    X = (X - mean) / std

    mean_out = mean.view(-1)
    std_out = std.view(-1)

    if as_vector:
        X = X.view(X.size(0), -1)

    if shuffle:
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(X.size(0), generator=g)
        X = X[indices]
        y = y[indices]

    unique_labels = torch.unique(y).tolist()
    num_classes = len(unique_labels)

    print("Label One-Hot Encoding Mapping:")
    for i, label in enumerate(unique_labels):
        one_hot_vec = [0] * num_classes
        one_hot_vec[i] = 1
        print(f"  {label} -> {one_hot_vec} -> Class {i}")

    y = torch.nn.functional.one_hot(y, num_classes=num_classes).float()

    # Move to device
    X = X.to(device)
    y = y.to(device)

    return X, y, mean_out, std_out


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
