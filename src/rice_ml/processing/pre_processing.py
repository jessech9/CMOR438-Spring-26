"""Data preprocessing utilities (scalers, encoders, splits)."""

from __future__ import annotations

import numpy as np

from .._base import BaseEstimator, check_array

__all__ = [
    "StandardScaler",
    "MinMaxScaler",
    "LabelEncoder",
    "train_test_split",
    "one_hot_encode",
]


class StandardScaler(BaseEstimator):
    """Standardize features to zero mean and unit variance."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X) -> "StandardScaler":
        X = check_array(X)
        self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1])
        if self.with_std:
            scale = X.std(axis=0)
            scale[scale == 0] = 1.0
            self.scale_ = scale
        else:
            self.scale_ = np.ones(X.shape[1])
        return self

    def transform(self, X) -> np.ndarray:
        self._check_is_fitted(["mean_", "scale_"])
        X = check_array(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X) -> np.ndarray:
        self._check_is_fitted(["mean_", "scale_"])
        X = check_array(X)
        return X * self.scale_ + self.mean_


class MinMaxScaler(BaseEstimator):
    """Rescale each feature to a given range (default ``[0, 1]``)."""

    def __init__(self, feature_range: tuple = (0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None

    def fit(self, X) -> "MinMaxScaler":
        X = check_array(X)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        return self

    def transform(self, X) -> np.ndarray:
        self._check_is_fitted(["data_min_", "data_max_"])
        X = check_array(X)
        denom = self.data_max_ - self.data_min_
        denom = np.where(denom == 0, 1.0, denom)
        scaled = (X - self.data_min_) / denom
        lo, hi = self.feature_range
        return scaled * (hi - lo) + lo

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


class LabelEncoder(BaseEstimator):
    """Map categorical labels to consecutive integers ``0..K-1``."""

    def __init__(self):
        self.classes_: np.ndarray | None = None

    def fit(self, y) -> "LabelEncoder":
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        return self

    def transform(self, y) -> np.ndarray:
        self._check_is_fitted(["classes_"])
        y = np.asarray(y)
        out = np.empty(len(y), dtype=int)
        lookup = {label: i for i, label in enumerate(self.classes_)}
        for i, label in enumerate(y):
            if label not in lookup:
                raise ValueError(f"Label {label!r} was not seen during fit().")
            out[i] = lookup[label]
        return out

    def fit_transform(self, y) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y) -> np.ndarray:
        self._check_is_fitted(["classes_"])
        y = np.asarray(y, dtype=int)
        return self.classes_[y]


def one_hot_encode(y, num_classes: int | None = None) -> np.ndarray:
    """Convert an integer label vector to a one-hot matrix."""
    y = np.asarray(y, dtype=int)
    if num_classes is None:
        num_classes = int(y.max()) + 1
    out = np.zeros((len(y), num_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out


def train_test_split(
    *arrays,
    test_size: float = 0.2,
    shuffle: bool = True,
    stratify=None,
    random_state: int | None = None,
):
    """Split arrays into random train and test subsets."""
    if len(arrays) == 0:
        raise ValueError("train_test_split requires at least one array.")
    n = len(arrays[0])
    for a in arrays:
        if len(a) != n:
            raise ValueError("All input arrays must have the same length.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in the open interval (0, 1).")

    rng = np.random.default_rng(random_state)
    arrs = [np.asarray(a) for a in arrays]

    if stratify is not None:
        strat = np.asarray(stratify)
        if len(strat) != n:
            raise ValueError("stratify must have the same length as the inputs.")
        train_idx_list, test_idx_list = [], []
        for cls in np.unique(strat):
            cls_idx = np.where(strat == cls)[0]
            if shuffle:
                cls_idx = rng.permutation(cls_idx)
            n_test = max(1, int(round(len(cls_idx) * test_size)))
            test_idx_list.append(cls_idx[:n_test])
            train_idx_list.append(cls_idx[n_test:])
        train_idx = np.concatenate(train_idx_list)
        test_idx = np.concatenate(test_idx_list)
        if shuffle:
            train_idx = rng.permutation(train_idx)
            test_idx = rng.permutation(test_idx)
    else:
        idx = rng.permutation(n) if shuffle else np.arange(n)
        n_test = max(1, int(round(n * test_size)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

    out = []
    for a in arrs:
        out.append(a[train_idx])
        out.append(a[test_idx])
    return out
