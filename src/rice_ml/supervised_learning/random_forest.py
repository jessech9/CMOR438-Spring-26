"""Random Forest: bagging + random feature subsets over decision trees."""

from __future__ import annotations

import numpy as np

from .._base import BaseEstimator, ClassifierMixin, RegressorMixin, check_array, check_X_y
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = ["RandomForestClassifier", "RandomForestRegressor"]


def _majority_vote(predictions: np.ndarray) -> np.ndarray:
    """Per-column majority vote on a (n_estimators, n_samples) matrix."""
    n_samples = predictions.shape[1]
    out = np.empty(n_samples, dtype=predictions.dtype)
    for j in range(n_samples):
        values, counts = np.unique(predictions[:, j], return_counts=True)
        out[j] = values[counts.argmax()]
    return out


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Random forest classifier: bagging + random feature subsets."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: int | float | str = "sqrt",
        random_state: int | None = None,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_: list[DecisionTreeClassifier] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X, y) -> "RandomForestClassifier":
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]

        self.estimators_ = []
        for _ in range(self.n_estimators):
            seed = int(rng.integers(0, 2**31 - 1))
            idx = np.random.default_rng(seed).integers(0, n, size=n)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = check_array(X)
        proba = np.zeros((X.shape[0], len(self.classes_)))
        for tree in self.estimators_:
            tree_classes = tree.classes_
            tree_proba = tree.predict_proba(X)
            for k, cls in enumerate(self.classes_):
                if cls in tree_classes:
                    j = int(np.where(tree_classes == cls)[0][0])
                    proba[:, k] += tree_proba[:, j]
        proba /= len(self.estimators_)
        return proba

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class RandomForestRegressor(BaseEstimator, RegressorMixin):
    """Random forest regressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: int | float | str = "sqrt",
        random_state: int | None = None,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_: list[DecisionTreeRegressor] = []

    def fit(self, X, y) -> "RandomForestRegressor":
        X, y = check_X_y(X, y, y_numeric=True)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]

        self.estimators_ = []
        for _ in range(self.n_estimators):
            seed = int(rng.integers(0, 2**31 - 1))
            idx = np.random.default_rng(seed).integers(0, n, size=n)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)
        return self

    def predict(self, X) -> np.ndarray:
        X = check_array(X)
        preds = np.array([t.predict(X) for t in self.estimators_])
        return preds.mean(axis=0)
