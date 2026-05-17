import numpy as np
import pytest
from sklearn import metrics as skm

from smartheart.metrics import (
    accuracy,
    auc,
    confusion_matrix,
    f1_score,
    precision,
    recall,
    roc_curve,
)


def _sample():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0])
    return y_true, y_pred


def test_accuracy_matches_sklearn():
    y_true, y_pred = _sample()
    assert accuracy(y_true, y_pred) == pytest.approx(skm.accuracy_score(y_true, y_pred))


def test_precision_matches_sklearn():
    y_true, y_pred = _sample()
    assert precision(y_true, y_pred) == pytest.approx(skm.precision_score(y_true, y_pred))


def test_recall_matches_sklearn():
    y_true, y_pred = _sample()
    assert recall(y_true, y_pred) == pytest.approx(skm.recall_score(y_true, y_pred))


def test_f1_matches_sklearn():
    y_true, y_pred = _sample()
    assert f1_score(y_true, y_pred) == pytest.approx(skm.f1_score(y_true, y_pred))


def test_confusion_matrix_shape_and_values():
    y_true, y_pred = _sample()
    cm = confusion_matrix(y_true, y_pred)
    sk_cm = skm.confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    assert np.array_equal(cm, sk_cm)


def test_precision_returns_zero_when_no_positive_predictions():
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 0, 0])
    assert precision(y_true, y_pred) == 0.0


def test_recall_returns_zero_when_no_actual_positives():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 1, 0])
    assert recall(y_true, y_pred) == 0.0


def test_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.uniform(size=200)
    fpr, tpr = roc_curve(y_true, y_proba)
    our_auc = auc(fpr, tpr)
    sk_auc = skm.roc_auc_score(y_true, y_proba)
    assert our_auc == pytest.approx(sk_auc, abs=1e-6)
