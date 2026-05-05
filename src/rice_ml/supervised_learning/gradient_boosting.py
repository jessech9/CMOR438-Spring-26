"""Gradient boosting for binary classification."""

from __future__ import annotations

import numpy as np

from .._base import BaseEstimator, ClassifierMixin, check_array, check_X_y
from .decision_tree import DecisionTreeRegressor

__all__ = ["GradientBoostingClassifier"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _count_split_features(node, counts: np.ndarray) -> None:
    if node is None or node.is_leaf():
        return
    counts[node.feature] += 1.0
    _count_split_features(node.left, counts)
    _count_split_features(node.right, counts)


class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """Binary gradient boosting classifier with shallow regression trees.

    Each tree is fit to the Bernoulli deviance pseudo-residuals
    ``y - sigmoid(F)`` and added to the current log-odds score.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self.init_: float | None = None
        self.estimators_: list[DecisionTreeRegressor] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X, y) -> "GradientBoostingClassifier":
        X, y = check_X_y(X, y)
        self.classes_, y_int = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("GradientBoostingClassifier supports binary targets only.")

        pos_rate = np.clip(y_int.mean(), 1e-12, 1.0 - 1e-12)
        self.init_ = float(np.log(pos_rate / (1.0 - pos_rate)))
        raw = np.full(X.shape[0], self.init_, dtype=float)

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            residual = y_int - _sigmoid(raw)
            seed = int(rng.integers(0, 2**31 - 1))
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=seed,
            )
            tree.fit(X, residual)
            raw += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)

        counts = np.zeros(X.shape[1], dtype=float)
        for tree in self.estimators_:
            _count_split_features(tree.tree_, counts)
        total = counts.sum()
        self.feature_importances_ = counts / total if total > 0 else counts
        return self

    def decision_function(self, X) -> np.ndarray:
        self._check_is_fitted(["init_", "feature_importances_"])
        X = check_array(X)
        raw = np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.estimators_:
            raw += self.learning_rate * tree.predict(X)
        return raw

    def predict_proba(self, X) -> np.ndarray:
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]
