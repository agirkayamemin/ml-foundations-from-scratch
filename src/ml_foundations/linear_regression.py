"""Linear regression trained with batch gradient descent."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_n_features, check_X, check_X_y
from .metrics import mean_squared_error, r2_score


class LinearRegression:
    """Ordinary linear regression optimized with batch gradient descent."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.01,
        max_iter: int = 1_000,
        tolerance: float = 1e-8,
        fit_intercept: bool = True,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if isinstance(max_iter, bool) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.fit_intercept = fit_intercept
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None
        self.loss_history_: list[float] = []
        self.n_iter_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearRegression:
        """Fit coefficients by minimizing mean squared error."""
        X_array, y_array = check_X_y(X, y)
        try:
            y_float = y_array.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("y must contain only numeric values.") from exc
        if not np.all(np.isfinite(y_float)):
            raise ValueError("y must contain only finite values.")

        n_samples, n_features = X_array.shape
        weights = np.zeros(n_features, dtype=float)
        intercept = 0.0
        self.loss_history_ = []

        for iteration in range(1, self.max_iter + 1):
            predictions = X_array @ weights + intercept
            errors = predictions - y_float
            gradient = (2.0 / n_samples) * (X_array.T @ errors)
            intercept_gradient = (
                2.0 * float(errors.mean()) if self.fit_intercept else 0.0
            )
            weights -= self.learning_rate * gradient
            intercept -= self.learning_rate * intercept_gradient

            loss = mean_squared_error(y_float, X_array @ weights + intercept)
            if not np.isfinite(loss):
                raise RuntimeError(
                    "Gradient descent diverged; try scaling X or lowering learning_rate."
                )
            self.loss_history_.append(loss)
            if iteration > 1 and abs(self.loss_history_[-2] - loss) <= self.tolerance:
                break

        self.coef_ = weights
        self.intercept_ = intercept
        self.n_iter_ = iteration
        self.n_features_in_ = n_features
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict continuous targets for *X*."""
        self._check_is_fitted()
        X_array = check_X(X)
        check_n_features(X_array, self.n_features_in_)  # type: ignore[arg-type]
        return X_array @ self.coef_ + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the coefficient of determination R²."""
        return r2_score(y, self.predict(X))

    def _check_is_fitted(self) -> None:
        if self.coef_ is None or self.intercept_ is None or self.n_features_in_ is None:
            raise RuntimeError("LinearRegression must be fitted before prediction.")
