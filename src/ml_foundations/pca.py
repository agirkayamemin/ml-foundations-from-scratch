"""Principal component analysis implemented with NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_n_features, check_X


class PCA:
    """Linear dimensionality reduction using covariance eigendecomposition."""

    def __init__(self, *, n_components: int | None = None) -> None:
        if n_components is not None and (
            isinstance(n_components, bool)
            or not isinstance(n_components, (int, np.integer))
            or n_components <= 0
        ):
            raise ValueError("n_components must be a positive integer or None.")
        self.n_components = None if n_components is None else int(n_components)
        self.mean_: NDArray[np.float64] | None = None
        self.components_: NDArray[np.float64] | None = None
        self.explained_variance_: NDArray[np.float64] | None = None
        self.explained_variance_ratio_: NDArray[np.float64] | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike) -> PCA:
        """Learn principal axes from centered observations."""
        X_array = check_X(X)
        n_samples, n_features = X_array.shape
        if n_samples < 2:
            raise ValueError("PCA requires at least two samples.")
        n_components = n_features if self.n_components is None else self.n_components
        if n_components > n_features:
            raise ValueError("n_components cannot exceed the number of features.")
        self.mean_ = X_array.mean(axis=0)
        centered = X_array - self.mean_
        covariance = (centered.T @ centered) / (n_samples - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        self.components_ = eigenvectors[:, :n_components].T
        self.explained_variance_ = eigenvalues[:n_components]
        total_variance = float(eigenvalues.sum())
        self.explained_variance_ratio_ = (
            eigenvalues[:n_components] / total_variance
            if total_variance > 0
            else np.zeros(n_components)
        )
        self.n_components_ = n_components
        self.n_features_in_ = n_features
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project observations onto the fitted principal axes."""
        self._check_is_fitted()
        X_array = check_X(X)
        check_n_features(X_array, self.n_features_in_)  # type: ignore[arg-type]
        return (X_array - self.mean_) @ self.components_.T

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Fit principal axes and project *X*."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Map projected data back into the original feature space."""
        self._check_is_fitted()
        X_array = check_X(X)
        check_n_features(X_array, self.n_components_)  # type: ignore[arg-type]
        return X_array @ self.components_ + self.mean_

    def _check_is_fitted(self) -> None:
        if self.mean_ is None or self.components_ is None or self.n_components_ is None:
            raise RuntimeError("PCA must be fitted before transformation.")
