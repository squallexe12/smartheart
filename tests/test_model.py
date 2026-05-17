import numpy as np
import pytest

from smartheart.model import sigmoid


def test_sigmoid_zero_is_half():
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


def test_sigmoid_large_positive_is_one():
    assert sigmoid(np.array([1e6]))[0] == pytest.approx(1.0)


def test_sigmoid_large_negative_is_zero():
    assert sigmoid(np.array([-1e6]))[0] == pytest.approx(0.0)


def test_sigmoid_no_overflow_warning():
    # Should not raise or warn even on extreme values
    out = sigmoid(np.array([1e9, -1e9, 0.0]))
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_sigmoid_vectorized_shape():
    x = np.linspace(-5, 5, 11)
    assert sigmoid(x).shape == x.shape


from smartheart.model import bce_loss


def test_bce_loss_perfect_prediction_is_zero():
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([1e-12, 1 - 1e-12, 1e-12, 1 - 1e-12])
    loss = bce_loss(y_true, y_pred)
    assert loss < 1e-9


def test_bce_loss_handles_zero_and_one_without_inf():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 1.0])
    loss = bce_loss(y_true, y_pred)
    assert np.isfinite(loss)


def test_bce_loss_higher_for_wrong_predictions():
    y_true = np.array([1.0, 1.0, 1.0])
    good = bce_loss(y_true, np.array([0.9, 0.9, 0.9]))
    bad = bce_loss(y_true, np.array([0.1, 0.1, 0.1]))
    assert bad > good
