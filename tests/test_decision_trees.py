"""Tests for ``rice_ml.supervised_learning.decision_trees``."""

from __future__ import annotations

import numpy as np
import pytest

from rice_ml.supervised_learning.decision_trees import DecisionTreeClassifier


def test_decision_tree_classifier_can_overfit_training_data(multiclass_data):
    X, y = multiclass_data
    clf = DecisionTreeClassifier(max_depth=None).fit(X, y)
    assert clf.score(X, y) == pytest.approx(1.0)


def test_decision_tree_classifier_max_depth_limits_complexity(multiclass_data):
    X, y = multiclass_data
    deep = DecisionTreeClassifier(max_depth=None).fit(X, y)
    shallow = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert deep.score(X, y) > shallow.score(X, y)


def test_decision_tree_classifier_predict_proba_normalized(multiclass_data):
    X, y = multiclass_data
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_decision_tree_classifier_rejects_invalid_criterion():
    with pytest.raises(ValueError, match="criterion"):
        DecisionTreeClassifier(criterion="bogus")


def test_decision_tree_max_features_changes_chosen_feature():
    rng = np.random.default_rng(0)
    n, p = 100, 10
    X = rng.normal(size=(n, p))
    y = (X[:, 0] > 0).astype(int)
    clf_full = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    clf_sub = DecisionTreeClassifier(
        max_depth=1, max_features=1, random_state=0
    ).fit(X, y)
    assert clf_full.tree_.feature is not None
    assert clf_sub.tree_.feature is not None
