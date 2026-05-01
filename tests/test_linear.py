"""Tests for the linear models in ``rice_ml.supervised_learning.linear``."""

from __future__ import annotations

import numpy as np
import pytest

from rice_ml.supervised_learning.linear import LinearRegression, LogisticRegression


def test_linear_regression_recovers_known_coefficients(regression_data):
    X, y, coef, intercept = regression_data
    model = LinearRegression().fit(X, y)
    np.testing.assert_allclose(model.coef_, coef, atol=0.05)
    assert model.intercept_ == pytest.approx(intercept, abs=0.05)
    assert model.score(X, y) > 0.95


def test_linear_regression_no_intercept():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    y = X @ np.array([1.0, -1.0])
    model = LinearRegression(fit_intercept=False).fit(X, y)
    assert model.intercept_ == 0.0
    np.testing.assert_allclose(model.coef_, [1.0, -1.0], atol=1e-10)


def test_logistic_regression_separates_well(binary_classification_data):
    X, y = binary_classification_data
    model = LogisticRegression(alpha=0.0, max_iter=2000).fit(X, y)
    assert model.score(X, y) > 0.95
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_logistic_regression_loss_is_monotonically_decreasing(binary_classification_data):
    X, y = binary_classification_data
    model = LogisticRegression(learning_rate=0.1, max_iter=200).fit(X, y)
    losses = np.array(model.loss_history_)
    assert losses[-1] <= losses[0]
    assert (np.diff(losses) <= 1e-8).mean() > 0.95


def test_logistic_regression_rejects_non_binary_labels():
    X = np.zeros((4, 2))
    y = np.array([0, 1, 2, 1])
    with pytest.raises(ValueError, match="binary labels"):
        LogisticRegression().fit(X, y)


def test_logistic_regression_predict_threshold_changes_predictions(binary_classification_data):
    X, y = binary_classification_data
    model = LogisticRegression(max_iter=500).fit(X, y)
    pred_low = model.predict(X, threshold=0.1)
    pred_high = model.predict(X, threshold=0.9)
    assert pred_low.sum() >= pred_high.sum()


def test_logistic_regression_alpha_must_be_nonnegative():
    with pytest.raises(ValueError, match="non-negative"):
        LogisticRegression(alpha=-1.0)