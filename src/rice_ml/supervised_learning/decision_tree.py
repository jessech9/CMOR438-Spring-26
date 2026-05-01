"""Decision trees: classifier (Gini / entropy) and regressor (MSE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .._base import BaseEstimator, ClassifierMixin, RegressorMixin, check_array, check_X_y

__all__ = ["DecisionTreeClassifier", "DecisionTreeRegressor"]


@dataclass
class _Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    value: Optional[np.ndarray] = None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _gini(p: np.ndarray) -> float:
    return float(1.0 - np.sum(p ** 2))


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """CART-style decision tree classifier."""

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = None,
        random_state: int | None = None,
    ):
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state

        self.tree_: _Node | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self._rng: np.random.Generator | None = None

    def _impurity(self, y: np.ndarray) -> float:
        counts = np.bincount(y, minlength=len(self.classes_))
        p = counts / counts.sum()
        return _gini(p) if self.criterion == "gini" else _entropy(p)

    def _resolve_max_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if mf == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError("max_features string must be 'sqrt' or 'log2'.")
        if isinstance(mf, float):
            return max(1, int(mf * n_features))
        return min(int(mf), n_features)

    def fit(self, X, y) -> "DecisionTreeClassifier":
        X, y = check_X_y(X, y)
        self.classes_, y_int = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.tree_ = self._grow(X, y_int, depth=0)
        return self

    def _leaf_proba(self, y_int: np.ndarray) -> np.ndarray:
        counts = np.bincount(y_int, minlength=len(self.classes_)).astype(float)
        return counts / counts.sum()

    def _grow(self, X: np.ndarray, y_int: np.ndarray, depth: int) -> _Node:
        proba = self._leaf_proba(y_int)
        if (
            len(np.unique(y_int)) == 1
            or len(y_int) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return _Node(value=proba)

        feature, threshold, left_mask = self._best_split(X, y_int)
        if feature is None:
            return _Node(value=proba)

        right_mask = ~left_mask
        return _Node(
            feature=feature,
            threshold=threshold,
            left=self._grow(X[left_mask], y_int[left_mask], depth + 1),
            right=self._grow(X[right_mask], y_int[right_mask], depth + 1),
            value=proba,
        )

    def _best_split(self, X: np.ndarray, y_int: np.ndarray):
        n, p = X.shape
        n_features = self._resolve_max_features(p)
        feat_idx = (
            self._rng.choice(p, size=n_features, replace=False)
            if n_features < p
            else np.arange(p)
        )

        parent_imp = self._impurity(y_int)
        best_gain = 0.0
        best_feat: int | None = None
        best_thresh: float | None = None
        best_mask: np.ndarray | None = None

        for j in feat_idx:
            col = X[:, j]
            unique_vals = np.unique(col)
            if len(unique_vals) < 2:
                continue
            midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            for t in midpoints:
                left = col <= t
                n_left = left.sum()
                n_right = n - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                gain = parent_imp - (
                    n_left / n * self._impurity(y_int[left])
                    + n_right / n * self._impurity(y_int[~left])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(j)
                    best_thresh = float(t)
                    best_mask = left

        return best_feat, best_thresh, best_mask

    def _traverse(self, x: np.ndarray, node: _Node) -> _Node:
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict_proba(self, X) -> np.ndarray:
        self._check_is_fitted(["tree_"])
        X = check_array(X)
        out = np.zeros((X.shape[0], len(self.classes_)))
        for i, x in enumerate(X):
            out[i] = self._traverse(x, self.tree_).value
        return out

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------

class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """CART-style regression tree using mean squared error."""

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = None,
        random_state: int | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state

        self.tree_: _Node | None = None
        self.n_features_in_: int | None = None
        self._rng: np.random.Generator | None = None

    def _resolve_max_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if mf == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError("max_features string must be 'sqrt' or 'log2'.")
        if isinstance(mf, float):
            return max(1, int(mf * n_features))
        return min(int(mf), n_features)

    def fit(self, X, y) -> "DecisionTreeRegressor":
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.tree_ = self._grow(X, y, depth=0)
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        if (
            len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or np.allclose(y, y[0])
        ):
            return _Node(value=np.array([float(y.mean())]))

        feature, threshold, left_mask = self._best_split(X, y)
        if feature is None:
            return _Node(value=np.array([float(y.mean())]))

        right_mask = ~left_mask
        return _Node(
            feature=feature,
            threshold=threshold,
            left=self._grow(X[left_mask], y[left_mask], depth + 1),
            right=self._grow(X[right_mask], y[right_mask], depth + 1),
            value=np.array([float(y.mean())]),
        )

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n, p = X.shape
        n_features = self._resolve_max_features(p)
        feat_idx = (
            self._rng.choice(p, size=n_features, replace=False)
            if n_features < p
            else np.arange(p)
        )

        parent_var = float(np.var(y) * len(y))
        best_reduction = 0.0
        best_feat: int | None = None
        best_thresh: float | None = None
        best_mask: np.ndarray | None = None

        for j in feat_idx:
            col = X[:, j]
            unique_vals = np.unique(col)
            if len(unique_vals) < 2:
                continue
            midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            for t in midpoints:
                left = col <= t
                n_left = left.sum()
                n_right = n - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                ssr = (
                    np.var(y[left]) * n_left + np.var(y[~left]) * n_right
                )
                reduction = parent_var - ssr
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feat = int(j)
                    best_thresh = float(t)
                    best_mask = left

        return best_feat, best_thresh, best_mask

    def _traverse(self, x: np.ndarray, node: _Node) -> _Node:
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict(self, X) -> np.ndarray:
        self._check_is_fitted(["tree_"])
        X = check_array(X)
        out = np.empty(X.shape[0])
        for i, x in enumerate(X):
            out[i] = self._traverse(x, self.tree_).value[0]
        return out
