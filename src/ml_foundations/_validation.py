"""Shared validation helpers for estimators."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def check_X(X: ArrayLike, *, allow_empty: bool = False) -> NDArray[np.float64]:
    """Return *X* as a finite two-dimensional floating-point array."""
    try:
        array = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must contain only numeric values.") from exc
    if array.ndim != 2:
        raise ValueError(f"X must be a 2D array; got {array.ndim} dimensions.")
    if not allow_empty and (array.shape[0] == 0 or array.shape[1] == 0):
        raise ValueError("X must contain at least one sample and one feature.")
    if not np.all(np.isfinite(array)):
        raise ValueError("X must contain only finite values.")
    return array


def check_X_y(X: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray]:
    """Validate feature and target arrays and return aligned arrays."""
    X_array = check_X(X)
    y_array = np.asarray(y)
    if y_array.ndim != 1:
        raise ValueError(f"y must be a 1D array; got {y_array.ndim} dimensions.")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if y_array.shape[0] == 0:
        raise ValueError("y must contain at least one sample.")
    return X_array, y_array


def check_n_features(X: NDArray[np.float64], expected: int) -> None:
    """Raise when *X* has a different feature count than training data."""
    if X.shape[1] != expected:
        raise ValueError(f"X must have {expected} features; got {X.shape[1]}.")
