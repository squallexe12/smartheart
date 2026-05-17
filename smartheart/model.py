"""Logistic regression implemented from scratch in NumPy."""
from __future__ import annotations

import numpy as np

_SIGMOID_CLIP = 500.0


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid. Clips inputs to avoid exp overflow."""
    z_clipped = np.clip(z, -_SIGMOID_CLIP, _SIGMOID_CLIP)
    return 1.0 / (1.0 + np.exp(-z_clipped))


_BCE_EPS = 1e-15


def bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary cross-entropy loss, averaged over samples."""
    y_pred = np.clip(y_pred, _BCE_EPS, 1.0 - _BCE_EPS)
    losses = y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
    return float(-np.mean(losses))


class LogisticRegressionScratch:
    """Logistic regression with vanilla batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient updates.
    n_iters : int
        Number of full-batch gradient descent iterations.
    random_state : int
        Seed for weight initialization.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iters: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.random_state = random_state
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0
        self._cost_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.random_state)
        self._weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self._bias = 0.0
        self._cost_history = []

        for _ in range(self.n_iters):
            z = X @ self._weights + self._bias
            y_pred = sigmoid(z)
            error = y_pred - y
            grad_w = (X.T @ error) / n_samples
            grad_b = float(np.mean(error))
            self._weights -= self.learning_rate * grad_w
            self._bias -= self.learning_rate * grad_b
            self._cost_history.append(bce_loss(y, y_pred))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Model must be fit before predict_proba().")
        X = np.asarray(X, dtype=np.float64)
        return sigmoid(X @ self._weights + self._bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    @property
    def cost_history(self) -> list[float]:
        return list(self._cost_history)

    @property
    def weights(self) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Model not fit yet.")
        return self._weights.copy()

    @property
    def bias(self) -> float:
        return self._bias
