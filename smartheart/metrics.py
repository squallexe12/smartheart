"""Manual classification metrics — NumPy only."""
from __future__ import annotations

import numpy as np


def _to_int_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _to_int_arrays(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _to_int_arrays(y_true, y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _to_int_arrays(y_true, y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Returns the 2x2 confusion matrix in sklearn orientation:
    rows = actual, columns = predicted; order [[TN, FP], [FN, TP]].
    """
    y_true, y_pred = _to_int_arrays(y_true, y_pred)
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) arrays sorted by ascending fpr; matches sklearn shape conventions."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    # Sort by descending probability
    order = np.argsort(-y_proba, kind="mergesort")
    y_sorted = y_true[order]
    p_sorted = y_proba[order]

    # Identify threshold change points (distinct probabilities)
    distinct_value_indices = np.where(np.diff(p_sorted))[0]
    threshold_idxs = np.concatenate([distinct_value_indices, [y_true.size - 1]])

    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    total_pos = tps[-1]
    total_neg = fps[-1]
    tpr = np.concatenate([[0.0], tps / total_pos]) if total_pos > 0 else np.zeros(len(tps) + 1)
    fpr = np.concatenate([[0.0], fps / total_neg]) if total_neg > 0 else np.zeros(len(fps) + 1)
    return fpr, tpr


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Trapezoidal AUC; expects fpr sorted ascending."""
    # NumPy 2.0 renamed np.trapz -> np.trapezoid; fall back for older versions.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(tpr, fpr))
