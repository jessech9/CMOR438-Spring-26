"""Feed-forward multilayer perceptron classifier."""

from __future__ import annotations

import numpy as np

from .._base import BaseEstimator, ClassifierMixin, check_array, check_X_y
from ..processing.pre_processing import one_hot_encode

__all__ = ["MLPClassifier"]


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _relu_grad(a: np.ndarray) -> np.ndarray:
    return (a > 0).astype(float)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLPClassifier(BaseEstimator, ClassifierMixin):
    r"""Feed-forward neural network classifier.

    Architecture: input -> (Linear -> ReLU)\* -> Linear -> softmax.
    Trained with mini-batch SGD, categorical cross-entropy, He init, and
    optional L2 weight decay (``alpha``).
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (32,),
        learning_rate: float = 0.05,
        max_iter: int = 200,
        batch_size: int = 32,
        alpha: float = 1e-4,
        random_state: int | None = None,
    ):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.alpha = float(alpha)
        self.random_state = random_state

        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.classes_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def _init_params(self, n_features: int, n_classes: int) -> None:
        rng = np.random.default_rng(self.random_state)
        sizes = [n_features, *self.hidden_layer_sizes, n_classes]
        self.weights_ = []
        self.biases_ = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            std = np.sqrt(2.0 / fan_in)
            self.weights_.append(rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.biases_.append(np.zeros(fan_out))

    def _forward(self, X: np.ndarray):
        activations = [X]
        pre_acts = []
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            Z = activations[-1] @ W + b
            pre_acts.append(Z)
            if i == len(self.weights_) - 1:
                activations.append(_softmax(Z))
            else:
                activations.append(_relu(Z))
        return activations, pre_acts

    def fit(self, X, y) -> "MLPClassifier":
        X, y = check_X_y(X, y)
        self.classes_, y_int = np.unique(y, return_inverse=True)
        Y = one_hot_encode(y_int, num_classes=len(self.classes_))
        self._init_params(X.shape[1], len(self.classes_))

        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        self.loss_history_ = []

        for _ in range(self.max_iter):
            order = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                Xb = X[batch]
                Yb = Y[batch]
                acts, _ = self._forward(Xb)
                proba = acts[-1]

                eps = 1e-12
                loss = -np.mean(np.sum(Yb * np.log(proba + eps), axis=1))
                if self.alpha > 0:
                    loss = loss + 0.5 * self.alpha * sum(
                        float(np.sum(W * W)) for W in self.weights_
                    )
                epoch_loss += float(loss) * len(batch)

                grads_W = [None] * len(self.weights_)
                grads_b = [None] * len(self.biases_)
                m = Xb.shape[0]
                delta = (proba - Yb) / m
                for layer in reversed(range(len(self.weights_))):
                    A_prev = acts[layer]
                    grads_W[layer] = A_prev.T @ delta + self.alpha * self.weights_[layer]
                    grads_b[layer] = delta.sum(axis=0)
                    if layer > 0:
                        delta = (delta @ self.weights_[layer].T) * _relu_grad(acts[layer])

                for layer in range(len(self.weights_)):
                    self.weights_[layer] -= self.learning_rate * grads_W[layer]
                    self.biases_[layer] -= self.learning_rate * grads_b[layer]

            self.loss_history_.append(epoch_loss / n)
        return self

    def predict_proba(self, X) -> np.ndarray:
        self._check_is_fitted(["weights_"])
        X = check_array(X)
        acts, _ = self._forward(X)
        return acts[-1]

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
