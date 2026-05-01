"""DBSCAN: Density-Based Spatial Clustering of Applications with Noise."""

from __future__ import annotations

import numpy as np

from .._base import BaseEstimator, ClusterMixin, check_array

__all__ = ["DBSCAN"]


class DBSCAN(BaseEstimator, ClusterMixin):
    """Density-Based Spatial Clustering of Applications with Noise."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean"):
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1.")
        if metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric must be 'euclidean' or 'manhattan'.")

        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.metric = metric
        self.labels_: np.ndarray | None = None
        self.core_sample_indices_: np.ndarray | None = None

    def _neighbors(self, X: np.ndarray, i: int) -> np.ndarray:
        if self.metric == "euclidean":
            d = np.linalg.norm(X - X[i], axis=1)
        else:
            d = np.abs(X - X[i]).sum(axis=1)
        return np.where(d <= self.eps)[0]

    def fit(self, X) -> "DBSCAN":
        X = check_array(X)
        n = X.shape[0]
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0
        core_indices = []

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._neighbors(X, i)
            if len(neighbors) < self.min_samples:
                continue

            labels[i] = cluster_id
            core_indices.append(i)
            seeds = list(neighbors)
            j = 0
            while j < len(seeds):
                q = seeds[j]
                if not visited[q]:
                    visited[q] = True
                    q_neighbors = self._neighbors(X, q)
                    if len(q_neighbors) >= self.min_samples:
                        core_indices.append(q)
                        for n_idx in q_neighbors:
                            if n_idx not in seeds:
                                seeds.append(n_idx)
                if labels[q] == -1:
                    labels[q] = cluster_id
                j += 1

            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.array(sorted(set(core_indices)), dtype=int)
        return self
