"""Rosenblatt's single-layer perceptron."""
from __future__ import annotations
import numpy as np
from .._base import BaseEstimator, ClassifierMixin, check_array, check_X_y
 
__all__ = ["Perceptron"]
 
 
class Perceptron(BaseEstimator, ClassifierMixin):
    """Binary classifier using Rosenblatt's perceptron learning rule.
 
    Iterates over training samples one at a time and nudges the weights
    whenever a prediction is wrong. Converges once a full epoch has zero
    mistakes, or when max_iter is reached. Labels must be 0 or 1.
    """
 
    def __init__(
        self,
        learning_rate: float = 1.0,
        max_iter: int = 1000,
        shuffle: bool = True,
        fit_intercept: bool = True,
        random_state: int | None = None,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
 
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.shuffle = bool(shuffle)
        self.fit_intercept = bool(fit_intercept)
        self.random_state = random_state
 
        # Fitted attributes — set by fit()
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_iter_: int = 0
        self.classes_: np.ndarray | None = None
 
    def fit(self, X, y) -> "Perceptron":
        """Train on X (features) and y (labels in {0, 1}). Returns self."""
        X, y = check_X_y(X, y)
        unique = np.unique(y)
        if not (len(unique) == 2 and set(unique.tolist()).issubset({0, 1})):
            raise ValueError("Perceptron only supports binary {0, 1} labels.")
 
        self.classes_ = np.array([0, 1])
        n, p = X.shape
        rng = np.random.default_rng(self.random_state)
 
        w = np.zeros(p)
        b = 0.0
 
        for epoch in range(1, self.max_iter + 1):
            order = rng.permutation(n) if self.shuffle else np.arange(n)
            mistakes = 0
            for i in order:
                z = X[i] @ w + b
                pred = 1 if z >= 0 else 0
                update = self.learning_rate * (y[i] - pred)
                if update != 0:
                    w = w + update * X[i]
                    if self.fit_intercept:
                        b = b + update
                    mistakes += 1
 
            if mistakes == 0:
                break
 
        self.coef_ = w
        self.intercept_ = float(b) if self.fit_intercept else 0.0
        self.n_iter_ = epoch
        return self
 
    def decision_function(self, X) -> np.ndarray:
        """Return the raw score (X @ w + b) for each sample. Positive = class 1."""
        self._check_is_fitted(["coef_"])
        X = check_array(X)
        return X @ self.coef_ + self.intercept_
 
    def predict(self, X) -> np.ndarray:
        """Return predicted labels (0 or 1) for each sample in X."""
        return (self.decision_function(X) >= 0).astype(int)
 