import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import torch
from typing import Callable


def load_iris():
    # TODO: Check if data is there/add error handling
    data = pd.read_csv('../data/iris/iris.data')

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



def split_data(X, y, seed, test_size=0.2):

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=test_size, random_state=seed)

    return {"X": X_train_full, "y": y_train_full}, {"X": X_train, "y": y_train}, {"X": X_val, "y": y_val}, {"X": X_test, "y": y_test}



# fixed_size = lambda total_size: 15

# proportion = lambda group_size: int(group_size * group_proportion)

def cluster_data(X, y, cluster_size_func: Callable[[int], int]):

    GROUP_PROPORTION = 0.25 
    # GROUP_PROPORTION = 0.5

    # Example data
    # X = np.random.randn(132, 2)

    original_shape = X.shape


    X_all_centroids = []
    y_all_centroids = []

    all_y = np.unique(y, axis=0)

    for selected_y in all_y:


        group = X[np.all(selected_y == y, axis=1)]
        group_size = len(group)

        group = group.reshape(group_size, -1)

        # Number of clusters
        k = cluster_size_func(len(group))

        print(f"{k} clusters for {selected_y}")

        # Create a KMeans object
        kmeans = KMeans(n_clusters=k, random_state=0)

        # Fit the model to the data and predict cluster assignments
        cluster_assignments = kmeans.fit_predict(group)

        # Get the centroids
        X_centroids_group = kmeans.cluster_centers_
        y_centroids_group = np.tile(selected_y, (k, 1))

        X_all_centroids.append(X_centroids_group)
        y_all_centroids.append(y_centroids_group)

    original_shape = list(original_shape)
    original_shape[0] = -1
    original_shape = tuple(original_shape)

    X_centroids =  np.concatenate(X_all_centroids).reshape(original_shape)
    y_centroids =  np.concatenate(y_all_centroids)
    return X_centroids, y_centroids