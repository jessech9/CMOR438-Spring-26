"""K-means clustering (Lloyd + k-means++)."""



from __future__ import annotations

import numpy as np

from .._base import BaseEstimator, ClusterMixin, check_array

__all__ = ["KMeans"]


# ---------------------------------------------------------------------------
# K-Means with k-means++ initialization
# ---------------------------------------------------------------------------

class KMeans(BaseEstimator, ClusterMixin):
    """K-means clustering using Lloyd's algorithm with k-means++ init."""

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        if init not in {"k-means++", "random"}:
            raise ValueError("init must be 'k-means++' or 'random'.")
        if n_init < 1:
            raise ValueError("n_init must be at least 1.")

        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    def _init_centers(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = X.shape[0]
        if self.init == "random":
            idx = rng.choice(n, size=self.n_clusters, replace=False)
            return X[idx].copy()

        first = int(rng.integers(0, n))
        centers = [X[first]]
        d2 = np.sum((X - centers[0]) ** 2, axis=1)
        for _ in range(1, self.n_clusters):
            probs = d2 / d2.sum()
            idx = int(rng.choice(n, p=probs))
            centers.append(X[idx])
            new_d2 = np.sum((X - centers[-1]) ** 2, axis=1)
            d2 = np.minimum(d2, new_d2)
        return np.asarray(centers)

    def _lloyd(self, X: np.ndarray, centers: np.ndarray, rng: np.random.Generator):
        prev_centers = centers.copy()
        for it in range(1, self.max_iter + 1):
            aa = (X * X).sum(axis=1)[:, None]
            cc = (centers * centers).sum(axis=1)[None, :]
            d2 = aa + cc - 2.0 * X @ centers.T
            np.maximum(d2, 0.0, out=d2)
            labels = d2.argmin(axis=1)

            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) == 0:
                    new_centers[k] = X[int(rng.integers(0, X.shape[0]))]
                else:
                    new_centers[k] = members.mean(axis=0)

            shift = np.linalg.norm(new_centers - prev_centers)
            centers = new_centers
            prev_centers = new_centers
            if shift < self.tol:
                break

        aa = (X * X).sum(axis=1)[:, None]
        cc = (centers * centers).sum(axis=1)[None, :]
        d2 = aa + cc - 2.0 * X @ centers.T
        np.maximum(d2, 0.0, out=d2)
        labels = d2.argmin(axis=1)
        inertia = float(d2[np.arange(X.shape[0]), labels].sum())
        return centers, labels, inertia, it

    def fit(self, X) -> "KMeans":
        X = check_array(X)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_clusters cannot exceed the number of samples.")
        rng = np.random.default_rng(self.random_state)

        best = None
        for _ in range(self.n_init):
            centers0 = self._init_centers(X, rng)
            centers, labels, inertia, n_iter = self._lloyd(X, centers0, rng)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia, n_iter)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X) -> np.ndarray:
        self._check_is_fitted(["cluster_centers_"])
        X = check_array(X)
        aa = (X * X).sum(axis=1)[:, None]
        cc = (self.cluster_centers_ * self.cluster_centers_).sum(axis=1)[None, :]
        d2 = aa + cc - 2.0 * X @ self.cluster_centers_.T
        return d2.argmin(axis=1)