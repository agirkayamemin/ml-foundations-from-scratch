"""K-Means clustering implemented with NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_n_features, check_X


class KMeans:
    """Partition observations into clusters by iterative centroid updates."""

    def __init__(
        self,
        *,
        n_clusters: int = 8,
        max_iter: int = 300,
        tolerance: float = 1e-4,
        random_state: int | None = None,
        n_init: int = 10,
    ) -> None:
        if isinstance(n_clusters, bool) or not isinstance(
            n_clusters, (int, np.integer)
        ):
            raise TypeError("n_clusters must be an integer.")
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        if isinstance(max_iter, bool) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        if (
            isinstance(n_init, bool)
            or not isinstance(n_init, (int, np.integer))
            or n_init <= 0
        ):
            raise ValueError("n_init must be a positive integer.")
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.random_state = random_state
        self.n_init = int(n_init)
        self.cluster_centers_: NDArray[np.float64] | None = None
        self.labels_: NDArray[np.int64] | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike) -> KMeans:
        """Compute cluster centers and assignments for *X*."""
        X_array = check_X(X)
        n_samples = X_array.shape[0]
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed the number of samples.")
        rng = np.random.default_rng(self.random_state)
        best: tuple[float, NDArray[np.float64], NDArray[np.int64], int] | None = None
        for _ in range(self.n_init):
            centers, labels, inertia, iteration = self._fit_once(X_array, rng)
            if best is None or inertia < best[0]:
                best = (inertia, centers, labels, iteration)
        assert best is not None
        self.inertia_, self.cluster_centers_, self.labels_, self.n_iter_ = best
        self.n_features_in_ = X_array.shape[1]
        return self

    def _fit_once(
        self, X: NDArray[np.float64], rng: np.random.Generator
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], float, int]:
        n_samples = X.shape[0]
        centers = X[rng.choice(n_samples, self.n_clusters, replace=False)].copy()
        for iteration in range(1, self.max_iter + 1):
            squared_distances = self._squared_distances(X, centers)
            labels = np.argmin(squared_distances, axis=1)
            new_centers = centers.copy()
            minimum_distances = squared_distances[np.arange(n_samples), labels]
            for cluster in range(self.n_clusters):
                members = X[labels == cluster]
                if members.size:
                    new_centers[cluster] = members.mean(axis=0)
                else:
                    farthest = int(np.argmax(minimum_distances))
                    new_centers[cluster] = X[farthest]
                    minimum_distances[farthest] = -1.0
            shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
            centers = new_centers
            if shift <= self.tolerance:
                break
        final_distances = self._squared_distances(X, centers)
        labels = np.argmin(final_distances, axis=1)
        inertia = float(np.sum(final_distances[np.arange(n_samples), labels]))
        return centers, labels, inertia, iteration

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Assign samples to the nearest fitted centroid."""
        self._check_is_fitted()
        X_array = check_X(X)
        check_n_features(X_array, self.n_features_in_)  # type: ignore[arg-type]
        return np.argmin(
            self._squared_distances(X_array, self.cluster_centers_), axis=1
        )

    def fit_predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Fit clusters and return training assignments."""
        return self.fit(X).labels_.copy()  # type: ignore[union-attr]

    @staticmethod
    def _squared_distances(
        X: NDArray[np.float64], centers: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)

    def _check_is_fitted(self) -> None:
        if self.cluster_centers_ is None or self.n_features_in_ is None:
            raise RuntimeError("KMeans must be fitted before prediction.")
