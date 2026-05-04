"""Tests for ``rice_ml.Unsupervised_Learning.svd``."""

from __future__ import annotations

import numpy as np
import pytest

from rice_ml.Unsupervised_Learning.svd import SVD


def test_svd_factorization_reconstructs_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    svd = SVD(n_components=6).fit(X)
    Z = svd.transform(X)
    X_rec = svd.inverse_transform(Z)
    np.testing.assert_allclose(X_rec, X, atol=1e-8)


def test_svd_top_components_are_orthonormal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    svd = SVD(n_components=3).fit(X)
    G = svd.components_ @ svd.components_.T
    np.testing.assert_allclose(G, np.eye(3), atol=1e-10)


def test_svd_singular_values_match_numpy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    svd = SVD(n_components=3).fit(X)
    expected = np.linalg.svd(X, compute_uv=False)[:3]
    np.testing.assert_allclose(svd.singular_values_, expected, atol=1e-10)


def test_svd_rejects_invalid_n_components():
    with pytest.raises(ValueError, match="positive integer"):
        SVD(n_components=0)