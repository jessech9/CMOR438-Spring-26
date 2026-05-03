"""Principal Component Analysis."""

from __future__ import annotations

import numpy as np

from .._base import BaseEstimator, check_array

__all__ = ["PCA"]


def _sign_flip(U: np.ndarray, Vt: np.ndarray):
    """Apply a deterministic sign convention so signs don't flip across runs."""
    max_abs_cols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_cols, range(U.shape[1])])
    signs = np.where(signs == 0, 1.0, signs)
    return U * signs, Vt * signs[:, None]


class PCA(BaseEstimator):
    """Principal Component Analysis via SVD on the centered data."""

    _estimator_type = "transformer"

    def __init__(self, n_components: int | None = None):
        if n_components is not None and n_components < 1:
            raise ValueError("n_components must be a positive integer or None.")
        self.n_components = n_components

        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X) -> "PCA":
        X = check_array(X)
        n_samples, n_features = X.shape
        max_components = min(n_samples, n_features)
        k = max_components if self.n_components is None else min(self.n_components, max_components)

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        U, Vt = _sign_flip(U, Vt)

        self.components_ = Vt[:k]
        self.singular_values_ = s[:k]
        self.explained_variance_ = (s[:k] ** 2) / max(n_samples - 1, 1)
        total_var = (s ** 2).sum() / max(n_samples - 1, 1)
        if total_var == 0:
            self.explained_variance_ratio_ = np.zeros(k)
        else:
            self.explained_variance_ratio_ = self.explained_variance_ / total_var
        return self

    def transform(self, X) -> np.ndarray:
        self._check_is_fitted(["components_", "mean_"])
        X = check_array(X)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z) -> np.ndarray:
        self._check_is_fitted(["components_", "mean_"])
        Z = check_array(Z)
        return Z @ self.components_ + self.mean_
