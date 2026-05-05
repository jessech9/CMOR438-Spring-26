"""Tests for gradient boosting classifiers."""

from __future__ import annotations

import numpy as np
import pytest

from rice_ml.supervised_learning.gradient_boosting import GradientBoostingClassifier


def test_gradient_boosting_classifier_fits_separable_binary_data(binary_classification_data):
    X, y = binary_classification_data
    model = GradientBoostingClassifier(
        n_estimators=40, learning_rate=0.2, max_depth=2, random_state=0
    ).fit(X, y)

    assert model.score(X, y) > 0.95


def test_gradient_boosting_predict_proba_shape_and_sum(binary_classification_data):
    X, y = binary_classification_data
    model = GradientBoostingClassifier(n_estimators=5, random_state=0).fit(X, y)
    proba = model.predict_proba(X[:7])

    assert proba.shape == (7, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(7))


def test_gradient_boosting_feature_importances_sum_to_one(binary_classification_data):
    X, y = binary_classification_data
    model = GradientBoostingClassifier(n_estimators=10, random_state=0).fit(X, y)

    assert model.feature_importances_.shape == (X.shape[1],)
    assert model.feature_importances_.sum() == pytest.approx(1.0)


def test_gradient_boosting_rejects_multiclass_targets(multiclass_data):
    X, y = multiclass_data
    with pytest.raises(ValueError, match="binary targets"):
        GradientBoostingClassifier(n_estimators=5).fit(X, y)


def test_gradient_boosting_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="n_estimators"):
        GradientBoostingClassifier(n_estimators=0)
    with pytest.raises(ValueError, match="learning_rate"):
        GradientBoostingClassifier(learning_rate=0)
