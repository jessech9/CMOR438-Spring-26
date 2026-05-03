"""Tests for ``rice_ml.Unsupervised_Learning.pca``."""

from __future__ import annotations

import numpy as np
import pytest

from rice_ml.Unsupervised_Learning.pca import PCA


def test_pca_recovers_principal_axis_in_2d():
    rng = np.random.default_rng(0)
    n = 500
    t = rng.normal(scale=3.0, size=n)
    X = np.column_stack([t, t]) + 0.05 * rng.normal(size=(n, 2))
    pca = PCA(n_components=2).fit(X)
    primary_axis = pca.components_[0] / np.linalg.norm(pca.components_[0])
    expected = np.array([1, 1]) / np.sqrt(2)
    cos_sim = abs(primary_axis @ expected)
    assert cos_sim > 0.99


def test_pca_explained_variance_ratio_sums_to_one_with_full_components():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    pca = PCA().fit(X)
    assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0, abs=1e-8)


def test_pca_inverse_transform_reconstructs_with_full_components():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    pca = PCA(n_components=5).fit(X)
    Z = pca.transform(X)
    np.testing.assert_allclose(pca.inverse_transform(Z), X, atol=1e-10)


def test_pca_n_components_capped_by_data_shape():
    X = np.zeros((10, 3))
    X[0, 0] = 1.0
    pca = PCA(n_components=20).fit(X)
    assert pca.components_.shape[0] == 3


def test_pca_rejects_invalid_n_components():
    with pytest.raises(ValueError, match="positive integer"):
        PCA(n_components=0)
