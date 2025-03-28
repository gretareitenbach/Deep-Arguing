from typing import Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

from deeparguing.clustering import Cluster


class kMeansCluster(Cluster):

    def __init__(self, cluster_size) -> None:
        super().__init__()
        self.cluster_size = cluster_size

    def cluster_data(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        X_all_centroids = []
        y_all_centroids = []

        all_y = np.unique(y_, axis=0)

        for selected_y in all_y:

            group = X_[np.all(selected_y == y_, axis=1)]
            group_size = len(group)

            group = group.reshape(group_size, -1)

            kmeans = KMeans(n_clusters=self.cluster_size, random_state=0)

            kmeans.fit_predict(group)

            X_centroids_group = kmeans.cluster_centers_
            y_centroids_group = np.tile(selected_y, (self.cluster_size, 1))

            X_all_centroids.append(X_centroids_group)
            y_all_centroids.append(y_centroids_group)

        X_centroids = np.concatenate(X_all_centroids, axis=0)
        y_centroids = np.concatenate(y_all_centroids, axis=0)

        X_centroids = torch.tensor(X_centroids, device=X.device, dtype=X.dtype)
        y_centroids = torch.tensor(y_centroids, device=X.device, dtype=X.dtype)

        return X_centroids, y_centroids
