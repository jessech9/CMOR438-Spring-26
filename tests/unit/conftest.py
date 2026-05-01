"""Shared fixtures used by unit tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def binary_classification_data(rng):
    """Two well-separated Gaussian clusters in 2D."""
    n_per = 80
    X1 = rng.normal(loc=[-2.0, -2.0], scale=0.8, size=(n_per, 2))
    X2 = rng.normal(loc=[+2.0, +2.0], scale=0.8, size=(n_per, 2))
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(n_per, dtype=int), np.ones(n_per, dtype=int)])
    return X, y


@pytest.fixture
def multiclass_data(rng):
    """Three Gaussian clusters in 2D for multi-class classification."""
    n_per = 60
    centers = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    X = np.vstack([c + rng.normal(scale=0.6, size=(n_per, 2)) for c in centers])
    y = np.repeat(np.arange(3), n_per)
    return X, y


@pytest.fixture
def cluster_data(rng):
    """Three well-separated 2D blobs for clustering tests."""
    n_per = 40
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    X = np.vstack([c + rng.normal(scale=0.4, size=(n_per, 2)) for c in centers])
    y = np.repeat(np.arange(3), n_per)
    return X, y
