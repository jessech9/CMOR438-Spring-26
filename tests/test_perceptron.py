"""Tests for ``rice_ml.supervised_learning.perceptron``."""
from __future__ import annotations
 
import numpy as np
import pytest
 
from rice_ml.supervised_learning.perceptron import Perceptron
 
 
def test_perceptron_learns_linearly_separable(binary_classification_data):
    """Perceptron should fit a cleanly separable dataset with high accuracy."""
    X, y = binary_classification_data
    clf = Perceptron(max_iter=100, random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.95
 
 
def test_perceptron_converges_early_on_separable_data(binary_classification_data):
    """Perceptron should stop before max_iter when data is linearly separable."""
    X, y = binary_classification_data
    clf = Perceptron(max_iter=1000, random_state=0).fit(X, y)
    assert clf.n_iter_ < 1000
 
 
def test_perceptron_rejects_multiclass_labels():
    """Perceptron should raise ValueError when labels are not binary {0, 1}."""
    X = np.ones((6, 2))
    y = np.array([0, 1, 2, 0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        Perceptron().fit(X, y)
 
 
def test_perceptron_rejects_invalid_learning_rate():
    """Perceptron should raise ValueError for non-positive learning_rate."""
    with pytest.raises(ValueError):
        Perceptron(learning_rate=0)
 
 
def test_perceptron_rejects_invalid_max_iter():
    """Perceptron should raise ValueError for max_iter less than 1."""
    with pytest.raises(ValueError):
        Perceptron(max_iter=0)
 
 
def test_perceptron_cannot_solve_xor():
    """Perceptron is a linear classifier and cannot learn XOR (non-linearly separable)."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    clf = Perceptron(max_iter=200, random_state=0).fit(X, y)
    assert clf.score(X, y) <= 0.75
 
 
def test_perceptron_no_intercept(binary_classification_data):
    """Perceptron with fit_intercept=False should set intercept_ to 0.0."""
    X, y = binary_classification_data
    clf = Perceptron(fit_intercept=False, random_state=0).fit(X, y)
    assert clf.intercept_ == 0.0
 
 
def test_perceptron_coef_shape(binary_classification_data):
    """coef_ should have one weight per feature after fitting."""
    X, y = binary_classification_data
    clf = Perceptron(random_state=0).fit(X, y)
    assert clf.coef_.shape == (X.shape[1],)
 
 
def test_perceptron_predict_values(binary_classification_data):
    """predict() should return only 0s and 1s."""
    X, y = binary_classification_data
    clf = Perceptron(random_state=0).fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset({0, 1})
 