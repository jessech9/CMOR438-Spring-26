"""Shared base classes and validation helpers."""

from __future__ import annotations

import inspect
from typing import Any, Iterable

import numpy as np


class NotFittedError(RuntimeError):
    """Raised when an estimator method is called before ``fit``."""


class BaseEstimator:
    """Lightweight mixin providing parameter introspection and a repr."""

    @classmethod
    def _param_names(cls) -> list[str]:
        sig = inspect.signature(cls.__init__)
        return [
            name
            for name, p in sig.parameters.items()
            if name != "self" and p.kind != inspect.Parameter.VAR_KEYWORD
        ]

    def get_params(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> "BaseEstimator":
        valid = set(self._param_names())
        for k, v in params.items():
            if k not in valid:
                raise ValueError(
                    f"Invalid parameter '{k}' for {self.__class__.__name__}. "
                    f"Valid parameters: {sorted(valid)}"
                )
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({items})"

    def _check_is_fitted(self, attrs: Iterable[str]) -> None:
        missing = [a for a in attrs if getattr(self, a, None) is None]
        if missing:
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted yet. "
                f"Call fit() before using this method."
            )


class ClassifierMixin:
    """Mixin that provides ``score`` as classification accuracy."""

    _estimator_type = "classifier"

    def score(self, X, y) -> float:
        from .processing.post_processing import accuracy_score

        return float(accuracy_score(y, self.predict(X)))


class RegressorMixin:
    """Mixin that provides ``score`` as the coefficient of determination R^2."""

    _estimator_type = "regressor"

    def score(self, X, y) -> float:
        from .processing.post_processing import r2_score

        return float(r2_score(y, self.predict(X)))


class ClusterMixin:
    """Mixin for unsupervised clustering algorithms."""

    _estimator_type = "clusterer"

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


def check_array(
    X,
    *,
    name: str = "X",
    ensure_2d: bool = True,
    dtype: type = float,
    allow_empty: bool = False,
) -> np.ndarray:
    """Convert ``X`` to a NumPy array with shape/dtype checks."""
    arr = np.asarray(X, dtype=dtype)
    if ensure_2d:
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must be 2D (got {arr.ndim}D with shape {arr.shape})."
            )
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def check_X_y(X, y, *, dtype: type = float, y_numeric: bool = False):
    """Validate a feature matrix ``X`` paired with a target vector ``y``."""
    X = check_array(X, name="X", dtype=dtype)
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1D (got shape {y_arr.shape}).")
    if X.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X and y have inconsistent number of samples: "
            f"{X.shape[0]} vs {y_arr.shape[0]}."
        )
    if y_numeric:
        y_arr = y_arr.astype(float)
    return X, y_arr
