import numpy as np
import pytest

from ml_foundations.preprocessing import StandardScaler, train_test_split


def test_standard_scaler_produces_zero_mean_and_unit_variance() -> None:
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    transformed = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-12)


def test_standard_scaler_handles_constant_feature_and_inverse() -> None:
    X = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    scaler = StandardScaler()
    transformed = scaler.fit_transform(X)
    np.testing.assert_allclose(transformed[:, 1], 0.0)
    np.testing.assert_allclose(scaler.inverse_transform(transformed), X)


def test_standard_scaler_rejects_transform_before_fit() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        StandardScaler().transform([[1.0]])


@pytest.mark.parametrize("bad_X", [[1, 2], [], [[1, np.nan]], [[1, np.inf]]])
def test_standard_scaler_rejects_invalid_input(bad_X: object) -> None:
    with pytest.raises(ValueError):
        StandardScaler().fit(bad_X)


def test_train_test_split_is_reproducible_and_aligned() -> None:
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    first = train_test_split(X, y, test_size=0.3, random_state=42)
    second = train_test_split(X, y, test_size=0.3, random_state=42)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    X_train, X_test, y_train, y_test = first
    np.testing.assert_array_equal(X_train[:, 0] // 2, y_train)
    np.testing.assert_array_equal(X_test[:, 0] // 2, y_test)


def test_train_test_split_without_shuffle_preserves_order() -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        np.arange(5), np.arange(5), test_size=2, shuffle=False
    )
    np.testing.assert_array_equal(X_test, [0, 1])
    np.testing.assert_array_equal(X_train, [2, 3, 4])
    np.testing.assert_array_equal(y_test, [0, 1])
    np.testing.assert_array_equal(y_train, [2, 3, 4])


@pytest.mark.parametrize("test_size", [0, 1.0, True, 5, "bad"])
def test_train_test_split_rejects_invalid_size(test_size: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        train_test_split([1, 2, 3, 4], [1, 2, 3, 4], test_size=test_size)
