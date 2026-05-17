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


from smartheart.model import LogisticRegressionScratch


def _toy_linearly_separable(n_per_class=50, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(loc=2.0, scale=0.5, size=(n_per_class, 2))
    neg = rng.normal(loc=-2.0, scale=0.5, size=(n_per_class, 2))
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n_per_class), np.zeros(n_per_class)])
    return X, y


def test_fit_returns_self():
    X, y = _toy_linearly_separable()
    model = LogisticRegressionScratch(n_iters=50)
    assert model.fit(X, y) is model


def test_fit_reduces_cost_monotonically_at_first():
    X, y = _toy_linearly_separable()
    model = LogisticRegressionScratch(learning_rate=0.1, n_iters=100).fit(X, y)
    cost = model.cost_history
    assert len(cost) == 100
    # First 20 iterations strictly decreasing on a separable problem with lr=0.1
    diffs = np.diff(cost[:20])
    assert np.all(diffs <= 0), f"Cost not decreasing: {diffs}"


def test_predict_proba_in_unit_interval():
    X, y = _toy_linearly_separable()
    model = LogisticRegressionScratch(n_iters=50).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0],)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_predict_returns_binary_labels():
    X, y = _toy_linearly_separable()
    model = LogisticRegressionScratch(n_iters=50).fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds).tolist()).issubset({0, 1})


def test_fit_achieves_high_accuracy_on_separable_data():
    X, y = _toy_linearly_separable()
    model = LogisticRegressionScratch(learning_rate=0.1, n_iters=500).fit(X, y)
    preds = model.predict(X)
    accuracy = float(np.mean(preds == y))
    assert accuracy > 0.95


def test_seed_reproducibility():
    X, y = _toy_linearly_separable()
    m1 = LogisticRegressionScratch(random_state=42, n_iters=50).fit(X, y)
    m2 = LogisticRegressionScratch(random_state=42, n_iters=50).fit(X, y)
    assert np.allclose(m1.weights, m2.weights)
    assert m1.bias == pytest.approx(m2.bias)


def test_weights_shape_matches_features():
    X, y = _toy_linearly_separable()
    model = LogisticRegressionScratch(n_iters=10).fit(X, y)
    assert model.weights.shape == (X.shape[1],)
