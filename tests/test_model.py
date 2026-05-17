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
