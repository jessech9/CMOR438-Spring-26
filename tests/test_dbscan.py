"""Tests for DBSCAN (rice_ml.unsupervised_learning.dbscan)."""

from __future__ import annotations

import numpy as np
import pytest

from rice_ml.unsupervised_learning.dbscan import DBSCAN


def _cluster_purity(true_labels, pred_labels) -> float:
    """Compute majority-label purity of a clustering (excluding noise)."""
    purity = 0
    for c in np.unique(pred_labels):
        if c == -1:
            continue
        members = true_labels[pred_labels == c]
        if len(members) == 0:
            continue
        _, counts = np.unique(members, return_counts=True)
        purity += counts.max()
    return purity / len(true_labels)


def test_dbscan_separates_three_blobs(cluster_data):
    X, y = cluster_data
    db = DBSCAN(eps=1.0, min_samples=4).fit(X)
    n_clusters = len(set(db.labels_) - {-1})
    assert n_clusters == 3
    assert _cluster_purity(y, db.labels_) > 0.9


def test_dbscan_marks_isolated_points_as_noise():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.05, -0.1],
            [10.0, 10.0],
        ]
    )
    db = DBSCAN(eps=0.5, min_samples=2).fit(X)
    assert db.labels_[3] == -1


def test_dbscan_validates_inputs():
    with pytest.raises(ValueError, match="eps"):
        DBSCAN(eps=0)
    with pytest.raises(ValueError, match="min_samples"):
        DBSCAN(min_samples=0)
