import numpy as np
import pytest

from smartheart.preprocessing import standardize_apply, standardize_fit


def test_standardize_fit_returns_mean_and_std():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    mean, std = standardize_fit(X)
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert mean == pytest.approx(np.array([2.0, 20.0]))


def test_standardize_apply_produces_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(200, 4))
    mean, std = standardize_fit(X)
    Z = standardize_apply(X, mean, std)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-6)


def test_standardize_apply_uses_supplied_statistics_not_input_stats():
    X_train = np.array([[0.0], [10.0]])
    mean, std = standardize_fit(X_train)
    X_test = np.array([[5.0]])
    Z = standardize_apply(X_test, mean, std)
    # mean=5, std=5 -> (5-5)/5 = 0
    assert Z[0, 0] == pytest.approx(0.0)


def test_standardize_zero_std_does_not_divide_by_zero():
    X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])  # column 0 constant
    mean, std = standardize_fit(X)
    Z = standardize_apply(X, mean, std)
    assert np.all(np.isfinite(Z))
