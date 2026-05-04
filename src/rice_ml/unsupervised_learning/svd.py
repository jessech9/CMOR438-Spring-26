"""Truncated Singular Value Decomposition."""

from __future__ import annotations

import numpy as np

from .._base import BaseEstimator, check_array

__all__ = ["SVD"]


def _sign_flip(U: np.ndarray, Vt: np.ndarray):
    max_abs_cols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_cols, range(U.shape[1])])
    signs = np.where(signs == 0, 1.0, signs)
    return U * signs, Vt * signs[:, None]


class SVD(BaseEstimator):
    r"""Truncated Singular Value Decomposition.

    Factorizes the data matrix (without centering) as
    :math:`X \approx U_k \Sigma_k V_k^\top`.∏
    """

    _estimator_type = "transformer"

    def __init__(self, n_components: int = 2):
        if n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        self.n_components = int(n_components)

        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X) -> "SVD":
        X = check_array(X)
        n_samples, n_features = X.shape
        k = min(self.n_components, n_samples, n_features)

        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        U, Vt = _sign_flip(U, Vt)

        self.components_ = Vt[:k]
        self.singular_values_ = s[:k]
        self.explained_variance_ = (s[:k] ** 2) / max(n_samples - 1, 1)
        total = (s ** 2).sum() / max(n_samples - 1, 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total if total > 0 else np.zeros(k)
        )
        return self

    def transform(self, X) -> np.ndarray:
        self._check_is_fitted(["components_"])
        X = check_array(X)
        return X @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z) -> np.ndarray:
        self._check_is_fitted(["components_"])
        Z = check_array(Z)
        return Z @ self.components_