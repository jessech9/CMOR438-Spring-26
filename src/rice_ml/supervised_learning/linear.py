"""Linear models: ordinary least squares and binary logistic regression."""

from __future__ import annotations

import numpy as np

from .._base import BaseEstimator, ClassifierMixin, RegressorMixin, check_array, check_X_y

__all__ = ["LinearRegression", "LogisticRegression"]


def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ---------------------------------------------------------------------------
# Ordinary Least Squares
# ---------------------------------------------------------------------------

class LinearRegression(BaseEstimator, RegressorMixin):
    """Ordinary Least Squares regression solved in closed form.

    The design matrix is solved with ``numpy.linalg.lstsq`` for numerical
    robustness when X is rank-deficient.

    Parameters
    ----------
    fit_intercept : bool, default True
        If True, an intercept column is added to the design matrix.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X, y) -> "LinearRegression":
        X, y = check_X_y(X, y, y_numeric=True)
        A = _add_bias(X) if self.fit_intercept else X
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w
        return self

    def predict(self, X) -> np.ndarray:
        self._check_is_fitted(["coef_"])
        X = check_array(X)
        return X @ self.coef_ + self.intercept_


# ---------------------------------------------------------------------------
# Logistic regression (binary)
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Binary logistic regression trained by full-batch gradient descent.

    Optionally adds an L2 penalty on the weights (``alpha``).
    """

    def __init__(
        self,
        alpha: float = 0.0,
        learning_rate: float = 0.1,
        max_iter: int = 2000,
        tol: float = 1e-6,
        fit_intercept: bool = True,
    ):
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = float(alpha)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.fit_intercept = fit_intercept

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.classes_: np.ndarray | None = None
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0

    @staticmethod
    def _check_binary(y: np.ndarray) -> None:
        unique = np.unique(y)
        if not (len(unique) == 2 and set(unique.tolist()).issubset({0, 1})):
            raise ValueError(
                "LogisticRegression supports binary labels in {0, 1}; "
                f"got {unique.tolist()}."
            )

    def fit(self, X, y) -> "LogisticRegression":
        X, y = check_X_y(X, y, y_numeric=True)
        self._check_binary(y)
        self.classes_ = np.array([0, 1])

        A = _add_bias(X) if self.fit_intercept else X
        n, d = A.shape
        w = np.zeros(d)
        prev_loss = np.inf
        self.loss_history_ = []

        for it in range(self.max_iter):
            z = A @ w
            p = _sigmoid(z)
            eps = 1e-12
            nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            if self.alpha > 0:
                w_pen = w.copy()
                if self.fit_intercept:
                    w_pen[0] = 0.0
                nll = nll + 0.5 * self.alpha * (w_pen @ w_pen)
            self.loss_history_.append(float(nll))

            grad = A.T @ (p - y) / n
            if self.alpha > 0:
                reg = self.alpha * w.copy()
                if self.fit_intercept:
                    reg[0] = 0.0
                grad = grad + reg

            w = w - self.learning_rate * grad
            if abs(prev_loss - nll) < self.tol:
                break
            prev_loss = nll

        self.n_iter_ = it + 1
        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w
        return self

    def decision_function(self, X) -> np.ndarray:
        self._check_is_fitted(["coef_"])
        X = check_array(X)
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        z = self.decision_function(X)
        p1 = _sigmoid(z)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)