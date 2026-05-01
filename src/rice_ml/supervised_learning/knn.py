"""k-Nearest Neighbors classifier and regressor."""

from __future__ import annotations

import numpy as np

from .._base import BaseEstimator, ClassifierMixin, RegressorMixin, check_array, check_X_y

__all__ = ["KNeighborsClassifier", "KNeighborsRegressor"]


def _pairwise_distances(A: np.ndarray, B: np.ndarray, *, metric: str) -> np.ndarray:
    if metric == "euclidean":
        aa = (A * A).sum(axis=1)[:, None]
        bb = (B * B).sum(axis=1)[None, :]
        d2 = aa + bb - 2.0 * A @ B.T
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2)
    if metric == "manhattan":
        return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
    if metric == "chebyshev":
        return np.abs(A[:, None, :] - B[None, :, :]).max(axis=2)
    raise ValueError(f"Unsupported metric '{metric}'.")


class _KNeighborsBase(BaseEstimator):
    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: str = "uniform",
        metric: str = "euclidean",
    ):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'.")
        if metric not in {"euclidean", "manhattan", "chebyshev"}:
            raise ValueError("metric must be 'euclidean', 'manhattan', or 'chebyshev'.")

        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.metric = metric
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def _kneighbors(self, X) -> tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted(["_X_train", "_y_train"])
        X = check_array(X)
        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError("Feature dimension does not match training data.")
        D = _pairwise_distances(X, self._X_train, metric=self.metric)
        idx = np.argpartition(D, self.n_neighbors - 1, axis=1)[:, : self.n_neighbors]
        sort_axis = np.take_along_axis(D, idx, axis=1).argsort(axis=1)
        idx = np.take_along_axis(idx, sort_axis, axis=1)
        d = np.take_along_axis(D, idx, axis=1)
        return d, idx

    @staticmethod
    def _vote_weights(distances: np.ndarray, scheme: str) -> np.ndarray:
        if scheme == "uniform":
            return np.ones_like(distances)
        eps = 1e-12
        zero_mask = distances <= eps
        weights = np.where(
            zero_mask.any(axis=1, keepdims=True),
            zero_mask.astype(float),
            1.0 / np.maximum(distances, eps),
        )
        return weights


class KNeighborsClassifier(_KNeighborsBase, ClassifierMixin):
    """k-Nearest Neighbors classifier."""

    def fit(self, X, y) -> "KNeighborsClassifier":
        X, y = check_X_y(X, y)
        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors cannot exceed the training-set size.")
        self._X_train = X
        self._y_train = y
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        d, idx = self._kneighbors(X)
        weights = self._vote_weights(d, self.weights)
        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes))
        class_to_col = {c: i for i, c in enumerate(self.classes_)}
        for i in range(X.shape[0]):
            for j in range(self.n_neighbors):
                col = class_to_col[self._y_train[idx[i, j]]]
                proba[i, col] += weights[i, j]
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


class KNeighborsRegressor(_KNeighborsBase, RegressorMixin):
    """k-Nearest Neighbors regressor."""

    def fit(self, X, y) -> "KNeighborsRegressor":
        X, y = check_X_y(X, y, y_numeric=True)
        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors cannot exceed the training-set size.")
        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X) -> np.ndarray:
        d, idx = self._kneighbors(X)
        weights = self._vote_weights(d, self.weights)
        y_neighbors = self._y_train[idx]
        return (weights * y_neighbors).sum(axis=1) / np.maximum(
            weights.sum(axis=1), 1e-12
        )
