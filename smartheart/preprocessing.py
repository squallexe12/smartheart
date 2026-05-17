"""Manual preprocessing utilities — NumPy only."""
from __future__ import annotations

import numpy as np

_STD_EPS = 1e-12


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-column mean and std from training data."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply (x - mean) / std using supplied statistics. Constant columns become zero."""
    X = np.asarray(X, dtype=np.float64)
    safe_std = np.where(std < _STD_EPS, 1.0, std)
    return (X - mean) / safe_std
