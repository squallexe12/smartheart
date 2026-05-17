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


from smartheart.preprocessing import train_test_split_manual


def test_split_sizes_match_test_size():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = np.array([0] * 50 + [1] * 50)
    X_train, X_test, y_train, y_test = train_test_split_manual(
        X, y, test_size=0.2, random_state=42
    )
    assert X_test.shape[0] == 20
    assert X_train.shape[0] == 80
    assert y_train.shape[0] == 80
    assert y_test.shape[0] == 20


def test_split_preserves_class_ratio_when_stratified():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = np.array([0] * 80 + [1] * 120)
    _, _, y_train, y_test = train_test_split_manual(
        X, y, test_size=0.25, random_state=0, stratify=True
    )
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    expected = 120 / 200
    assert abs(train_ratio - expected) < 0.05
    assert abs(test_ratio - expected) < 0.05


def test_split_is_reproducible_with_same_seed():
    X = np.arange(50).reshape(50, 1).astype(float)
    y = (np.arange(50) % 2).astype(int)
    a = train_test_split_manual(X, y, test_size=0.2, random_state=7)
    b = train_test_split_manual(X, y, test_size=0.2, random_state=7)
    for arr_a, arr_b in zip(a, b):
        assert np.array_equal(arr_a, arr_b)


def test_split_no_overlap_between_train_and_test():
    X = np.arange(100).reshape(100, 1).astype(float)
    y = (np.arange(100) % 2).astype(int)
    X_train, X_test, _, _ = train_test_split_manual(X, y, test_size=0.3, random_state=1)
    train_ids = set(X_train.flatten().tolist())
    test_ids = set(X_test.flatten().tolist())
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == 100


from smartheart.preprocessing import one_hot_encode


def test_one_hot_basic_shape_and_columns():
    values = np.array([0, 1, 2, 0, 1])
    encoded, columns = one_hot_encode(values, categories=[0, 1, 2], drop_first=False, prefix="cp")
    assert encoded.shape == (5, 3)
    assert columns == ["cp_0", "cp_1", "cp_2"]


def test_one_hot_drop_first_removes_one_column():
    values = np.array([0, 1, 2])
    encoded, columns = one_hot_encode(values, categories=[0, 1, 2], drop_first=True, prefix="cp")
    assert encoded.shape == (3, 2)
    assert columns == ["cp_1", "cp_2"]


def test_one_hot_correct_content():
    values = np.array([0, 1, 2])
    encoded, _ = one_hot_encode(values, categories=[0, 1, 2], drop_first=False, prefix="x")
    expected = np.eye(3)
    assert np.array_equal(encoded, expected)


def test_one_hot_uses_supplied_categories_for_unseen_values():
    values = np.array([0, 0, 0])
    encoded, columns = one_hot_encode(values, categories=[0, 1, 2], drop_first=False, prefix="x")
    assert encoded.shape == (3, 3)
    assert np.array_equal(encoded[:, 0], np.ones(3))
    assert np.array_equal(encoded[:, 1], np.zeros(3))
