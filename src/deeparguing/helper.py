import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def exclude_fields(X, excluded_indices):
    all_indices = np.arange(X.shape[-1])
    mask = np.logical_not(np.isin(all_indices, excluded_indices))
    return X[:, mask]


def one_hot_encode(data):
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(data)
    return encoder.transform(data)


def load_tabular_data(
    path,
    target_field,
    device="cpu",
    labels=[],
    size=-1,
    shuffle=False,
    seed=42,
    excluded_fields=[],
    one_hot_X=False,
):
    data = pd.read_csv(path, header=0)

    data = data.values
    X = exclude_fields(
        np.array(data), [target_field] + excluded_fields
    )
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


def load_mnist(as_vector=False, size=-1, labels=[]):

    from torchvision.datasets import MNIST

    mnist_trainset = MNIST("./temp/", train=True, download=True)
    # mnist_testset = MNIST("./temp/", train=False, download=True)

    X = mnist_trainset.data[:size].float().numpy()
    if as_vector:
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    y = mnist_trainset.targets[:size].numpy()

    if len(labels) != 0:
        mask = np.isin(y, labels)
        y = y[mask]
        X = X[mask]

    y = y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    # encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    return X, y


def split_data(X, y, seed, test_size=0.2):

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=test_size, random_state=seed
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(X, mean, std):
    return (X - mean) / std
