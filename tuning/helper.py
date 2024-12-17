import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def load_iris():
    # TODO: Check if data is there/add error handling
    data = pd.read_csv(parent_dir + '/data/iris/iris.data')

    data = data.values

    X = np.array(data[:, :-1], dtype=np.float32)
    y = np.array(data[:, -1])

    y = y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(y)
    y = encoder.transform(y)

    return X, y



def load_mnist(device="cpu"):

    from torchvision.datasets import MNIST
    mnist_trainset = MNIST("./temp/", train=True, download=True)
    # mnist_testset = MNIST("./temp/", train=False, download=True)


    X = mnist_trainset.data[:1000].float().to(device)
    y = mnist_trainset.targets[:1000].to(device)


    y = y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    # encoder = LabelEncoder()
    encoder.fit(y.cpu())
    y = encoder.transform(y.cpu())

    y = torch.tensor(y, dtype=torch.float32).to(device)

    return X, y

def load_glioma():
    # TODO: Check if data is there/add error handling
    data = pd.read_csv(parent_dir + '/data/glioma/TCGA_InfoWithGrade.csv')
    data = data.values

    X = np.array(data[:, 1:], dtype=np.float32)
    y = np.array(data[:, 0])

    y = y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(y)
    y = encoder.transform(y)

    return X, y

def load_covertype():
    # TODO: Check if data is there/add error handling
    data = pd.read_csv(parent_dir + '/data/covertype/covtype.data')
    data = data.values

    X = np.array(data[:, :-1], dtype=np.float32)
    y = np.array(data[:, -1])

    y = y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(y)
    y = encoder.transform(y)

    return X, y


def split_data(X, y, seed, test_size=0.2):

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=test_size, random_state=seed)

    return {"X": X_train_full, "y": y_train_full}, {"X": X_train, "y": y_train}, {"X": X_val, "y": y_val}, {"X": X_test, "y": y_test}



loaders = {
    "iris": load_iris,
    "mnist": load_mnist,
    "glioma": load_glioma,
    "covertype": load_covertype
}

def load_dataset(dataset):

    if dataset not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets are: {list(loaders.keys())}")

    return loaders[dataset]()



def normalise_input(X, mean, std):
    return (X-mean)/std




def to_torch(data, device="cpu"):
    X = data["X"]
    y = data["y"]

    X = torch.tensor(X, device=device, dtype=torch.float32)
    y = torch.tensor(y, device=device, dtype=torch.float32)

    return {"X": X, "y": y}