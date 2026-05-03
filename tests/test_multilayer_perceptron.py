"""Tests for ``rice_ml.supervised_learning.multilayer_perceptron``."""

from __future__ import annotations

import numpy as np

from rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier
from rice_ml.supervised_learning.perceptron import Perceptron


def test_mlp_solves_xor_with_hidden_layer():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    p = Perceptron(max_iter=200, random_state=0).fit(X, y)
    assert p.score(X, y) < 1.0
    mlp = MLPClassifier(
        hidden_layer_sizes=(8,),
        learning_rate=0.1,
        max_iter=2000,
        batch_size=4,
        random_state=0,
    ).fit(X, y)
    assert mlp.score(X, y) >= 0.75


def test_mlp_multiclass_classification(multiclass_data):
    X, y = multiclass_data
    mlp = MLPClassifier(
        hidden_layer_sizes=(16,),
        learning_rate=0.05,
        max_iter=200,
        batch_size=32,
        random_state=0,
    ).fit(X, y)
    assert mlp.score(X, y) > 0.9
    proba = mlp.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_mlp_loss_history_decreases(multiclass_data):
    X, y = multiclass_data
    mlp = MLPClassifier(
        hidden_layer_sizes=(8,), max_iter=100, random_state=0
    ).fit(X, y)
    losses = mlp.loss_history_
    assert losses[-1] < losses[0]
