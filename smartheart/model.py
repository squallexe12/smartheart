"""Logistic regression implemented from scratch in NumPy."""
from __future__ import annotations

import numpy as np

_SIGMOID_CLIP = 500.0


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid. Clips inputs to avoid exp overflow."""
    z_clipped = np.clip(z, -_SIGMOID_CLIP, _SIGMOID_CLIP)
    return 1.0 / (1.0 + np.exp(-z_clipped))
