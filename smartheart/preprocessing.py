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


def train_test_split_manual(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified or random train/test split.

    Returns (X_train, X_test, y_train, y_test).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    if not stratify:
        indices = rng.permutation(n)
        n_test = int(round(n * test_size))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
    else:
        train_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            shuffled = rng.permutation(cls_idx)
            n_test_cls = int(round(len(shuffled) * test_size))
            test_parts.append(shuffled[:n_test_cls])
            train_parts.append(shuffled[n_test_cls:])
        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
        # Re-shuffle so classes are interleaved
        train_idx = rng.permutation(train_idx)
        test_idx = rng.permutation(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
