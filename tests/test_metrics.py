import numpy as np
import pytest

from ml_foundations.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)


def test_regression_metrics() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 4.0]
    assert mean_squared_error(y_true, y_pred) == pytest.approx(1 / 3)
    assert r2_score(y_true, y_pred) == pytest.approx(0.5)


def test_r2_constant_target_policy() -> None:
    assert r2_score([2, 2], [2, 2]) == 1.0
    assert r2_score([2, 2], [1, 2]) == 0.0


def test_classification_metrics() -> None:
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 2])
    assert accuracy_score(y_true, y_pred) == 0.75
    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred),
        [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
    )


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        accuracy_score([1], [1, 2])
