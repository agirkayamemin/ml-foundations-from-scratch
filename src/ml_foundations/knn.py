"""k-nearest neighbors classification implemented with NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_n_features, check_X, check_X_y
from .metrics import accuracy_score


class KNeighborsClassifier:
    """Classify samples by deterministic majority vote among nearest neighbors."""

    def __init__(self, *, n_neighbors: int = 5) -> None:
        if isinstance(n_neighbors, bool) or not isinstance(
            n_neighbors, (int, np.integer)
        ):
            raise TypeError("n_neighbors must be an integer.")
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")
        self.n_neighbors = int(n_neighbors)
        self.X_train_: NDArray[np.float64] | None = None
        self.y_train_: NDArray | None = None
        self.classes_: NDArray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> KNeighborsClassifier:
        """Store training samples and their labels."""
        X_array, y_array = check_X_y(X, y)
        if self.n_neighbors > X_array.shape[0]:
            raise ValueError(
                "n_neighbors cannot exceed the number of training samples."
            )
        self.X_train_ = X_array.copy()
        self.y_train_ = y_array.copy()
        self.classes_ = np.unique(y_array)
        self.n_features_in_ = X_array.shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict labels using Euclidean distance and majority voting."""
        self._check_is_fitted()
        X_array = check_X(X)
        check_n_features(X_array, self.n_features_in_)  # type: ignore[arg-type]
        distances = np.sqrt(
            np.sum(
                (X_array[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]) ** 2,
                axis=2,
            )
        )
        neighbor_indices = np.argsort(distances, axis=1, kind="stable")[
            :, : self.n_neighbors
        ]
        predictions = []
        for indices in neighbor_indices:
            labels = self.y_train_[indices]
            counts = np.array(
                [np.count_nonzero(labels == label) for label in self.classes_]
            )
            predictions.append(self.classes_[int(np.argmax(counts))])
        return np.asarray(predictions, dtype=self.y_train_.dtype)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy."""
        return accuracy_score(y, self.predict(X))

    def _check_is_fitted(self) -> None:
        if self.X_train_ is None or self.y_train_ is None or self.classes_ is None:
            raise RuntimeError("KNeighborsClassifier must be fitted before prediction.")
