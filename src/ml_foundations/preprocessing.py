"""Data preprocessing utilities implemented with NumPy."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_n_features, check_X


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""

    def __init__(self) -> None:
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None
        self.var_: NDArray[np.float64] | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike) -> StandardScaler:
        """Learn per-feature mean and population variance from *X*."""
        X_array = check_X(X)
        self.mean_ = X_array.mean(axis=0)
        self.var_ = X_array.var(axis=0)
        raw_scale = np.sqrt(self.var_)
        self.scale_ = np.where(raw_scale == 0.0, 1.0, raw_scale)
        self.n_features_in_ = X_array.shape[1]
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Standardize *X* using fitted statistics."""
        self._check_is_fitted()
        X_array = check_X(X)
        check_n_features(X_array, self.n_features_in_)  # type: ignore[arg-type]
        return (X_array - self.mean_) / self.scale_

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Fit the scaler and standardize *X*."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Undo standardization using fitted statistics."""
        self._check_is_fitted()
        X_array = check_X(X)
        check_n_features(X_array, self.n_features_in_)  # type: ignore[arg-type]
        return X_array * self.scale_ + self.mean_

    def _check_is_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None or self.n_features_in_ is None:
            raise RuntimeError("StandardScaler must be fitted before transformation.")


def train_test_split(
    X: ArrayLike,
    y: ArrayLike,
    *,
    test_size: float = 0.25,
    random_state: int | None = None,
    shuffle: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Split aligned arrays into reproducible train and test subsets."""
    X_array = np.asarray(X)
    y_array = np.asarray(y)
    if X_array.ndim == 0 or y_array.ndim == 0:
        raise ValueError("X and y must be array-like collections.")
    n_samples = X_array.shape[0]
    if n_samples != y_array.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if n_samples < 2:
        raise ValueError("At least two samples are required for a split.")

    if isinstance(test_size, (bool, np.bool_)):
        raise TypeError("test_size must be a float or integer, not bool.")
    if isinstance(test_size, Integral):
        n_test = int(test_size)
    elif isinstance(test_size, Real):
        if not 0.0 < float(test_size) < 1.0:
            raise ValueError("Float test_size must be between 0 and 1.")
        n_test = int(np.ceil(n_samples * float(test_size)))
    else:
        raise TypeError("test_size must be a float or integer.")
    if not 1 <= n_test < n_samples:
        raise ValueError("test_size must leave at least one train and test sample.")

    indices = np.arange(n_samples)
    if shuffle:
        indices = np.random.default_rng(random_state).permutation(indices)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return (
        X_array[train_indices],
        X_array[test_indices],
        y_array[train_indices],
        y_array[test_indices],
    )
