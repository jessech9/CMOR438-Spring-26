"""Tests for k-nearest neighbors (rice_ml.supervised_learning.knn)."""

from __future__ import annotations

import numpy as np
import pytest

from rice_ml.supervised_learning.knn import KNeighborsClassifier, KNeighborsRegressor


def test_knn_classifier_perfect_on_separated_blobs(binary_classification_data):
    X, y = binary_classification_data
    clf = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    assert clf.score(X, y) > 0.95


def test_knn_classifier_predict_proba_sums_to_one(binary_classification_data):
    X, y = binary_classification_data
    proba = KNeighborsClassifier(n_neighbors=5).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_knn_classifier_distance_weights_change_predictions():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] > 0).astype(int)
    uniform = KNeighborsClassifier(n_neighbors=7, weights="uniform").fit(X, y)
    distance = KNeighborsClassifier(n_neighbors=7, weights="distance").fit(X, y)
    # Both should be reasonable; distance-weighted should not be worse on
    # this easy problem.
    assert uniform.score(X, y) >= 0.85
    assert distance.score(X, y) >= 0.85


def test_knn_regressor_learns_linear_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(150, 1))
    y = (2 * X[:, 0] + 1)
    reg = KNeighborsRegressor(n_neighbors=3).fit(X, y)
    test = np.linspace(-2.5, 2.5, 20).reshape(-1, 1)
    pred = reg.predict(test)
    truth = 2 * test[:, 0] + 1
    assert np.mean(np.abs(pred - truth)) < 0.4


def test_knn_supports_alternative_metrics(binary_classification_data):
    X, y = binary_classification_data
    for metric in ("euclidean", "manhattan", "chebyshev"):
        clf = KNeighborsClassifier(n_neighbors=5, metric=metric).fit(X, y)
        assert clf.score(X, y) > 0.85


def test_knn_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="positive integer"):
        KNeighborsClassifier(n_neighbors=0)
    with pytest.raises(ValueError, match="weights"):
        KNeighborsClassifier(weights="weighted")
    with pytest.raises(ValueError, match="metric"):
        KNeighborsClassifier(metric="cosine")


def test_knn_predict_rejects_feature_mismatch(binary_classification_data):
    X, y = binary_classification_data
    clf = KNeighborsClassifier().fit(X, y)
    with pytest.raises(ValueError, match="Feature dimension"):
        clf.predict(np.zeros((3, 5)))


def test_knn_n_neighbors_too_large():
    X = np.zeros((4, 2))
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="exceed"):
        KNeighborsClassifier(n_neighbors=10).fit(X, y)
