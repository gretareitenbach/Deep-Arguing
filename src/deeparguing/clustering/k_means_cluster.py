import logging
from typing import Tuple, override

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from torch import Tensor

from deeparguing.clustering import Cluster


class kMeansCluster(Cluster):

    def __init__(
        self, cluster_size: int | list[int], nearest_sample: bool = False
    ) -> None:
        super().__init__()
        self.cluster_size = cluster_size
        self.nearest_sample = nearest_sample

    @override
    def cluster_data(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        For each label in y, make clusters of X. The number of clusters is
        dependent on the cluster_size_func

        Parameters
        ----------

        X : array_like
            The inputs to be clustered

        y : array_like
            The labels of the inputs

        cluster_size_func : Callable[[int], int]
            A function that accepts the total number of items to be clustered
            and returns the number of clusters to produce


        Returns
        -------
        results : Tuple
            returns a pair of array_likes, the first of which is the cluster
            centers and the second is the corresponding label for each cluster
            center

        """

        X_ = X.cpu().numpy()
        y_ = y.cpu().numpy()

        original_shape = list(X_.shape)

        X_all_centroids = []
        y_all_centroids = []

        all_y = np.unique(y_, axis=0)

        for i, selected_y in enumerate(all_y):

            group = X_[np.all(selected_y == y_, axis=1)]
            group_size = len(group)

            group = group.reshape(group_size, -1)

            if isinstance(self.cluster_size, int):
                cluster_size = self.cluster_size
            else:
                cluster_size = self.cluster_size[i]
                logging.debug(
                    f"Class {i}, group_size: {group_size}, cluster_size: {cluster_size}"
                )

            kmeans = KMeans(n_clusters=cluster_size, random_state=0)

            kmeans.fit_predict(group)

            X_centroids_group = kmeans.cluster_centers_

            if self.nearest_sample:
                nearest_indices = pairwise_distances_argmin(X_centroids_group, group)
                X_centroids_group = group[nearest_indices]

            y_centroids_group = np.tile(selected_y, (cluster_size, 1))

            X_all_centroids.append(X_centroids_group)
            y_all_centroids.append(y_centroids_group)

        X_centroids = np.concatenate(X_all_centroids, axis=0)
        y_centroids = np.concatenate(y_all_centroids, axis=0)

        X_centroids = torch.tensor(X_centroids, device=X.device, dtype=X.dtype)
        original_shape[0] = -1
        X_centroids = X_centroids.reshape(original_shape)
        y_centroids = torch.tensor(y_centroids, device=X.device, dtype=X.dtype)

        return X_centroids, y_centroids
