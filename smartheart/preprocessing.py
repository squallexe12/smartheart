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


def one_hot_encode(
    values: np.ndarray,
    categories: list[int] | list[str],
    drop_first: bool = False,
    prefix: str = "cat",
) -> tuple[np.ndarray, list[str]]:
    """One-hot encode an integer/string array using a fixed category list.

    Parameters
    ----------
    values : np.ndarray
        1-D array of category values.
    categories : list
        Ordered list of possible categories. Values not in this list map to all-zero rows.
    drop_first : bool
        If True, omit the first category column (avoids collinearity).
    prefix : str
        Column name prefix; columns are ``f"{prefix}_{category}"``.
    """
    values = np.asarray(values).reshape(-1)
    used = categories[1:] if drop_first else categories
    encoded = np.zeros((values.shape[0], len(used)), dtype=np.float64)
    for col_idx, cat in enumerate(used):
        encoded[:, col_idx] = (values == cat).astype(np.float64)
    columns = [f"{prefix}_{cat}" for cat in used]
    return encoded, columns
