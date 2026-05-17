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
