"""Tests for random forest (rice_ml.supervised_learning.random_forest)."""

from __future__ import annotations

import numpy as np
import pytest

from rice_ml.supervised_learning.random_forest import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from rice_ml.supervised_learning.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


def test_random_forest_classifier_beats_single_stump(multiclass_data):
    X, y = multiclass_data
    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    rf = RandomForestClassifier(
        n_estimators=25, max_depth=3, random_state=0
    ).fit(X, y)
    assert rf.score(X, y) >= stump.score(X, y)


def test_random_forest_regressor_outperforms_single_tree():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3.0, 3.0, size=(150, 1))
    y = (np.sin(X[:, 0]) + 0.05 * rng.normal(size=150))
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    rf = RandomForestRegressor(
        n_estimators=20, max_depth=3, random_state=0
    ).fit(X, y)
    assert rf.score(X, y) >= tree.score(X, y) - 0.05
