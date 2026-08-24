import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from ml_foundations.linear_regression import LinearRegression
from ml_foundations.preprocessing import StandardScaler


def test_linear_regression_fits_single_feature() -> None:
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 3.0 * X[:, 0] + 2.0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression(learning_rate=0.1, max_iter=5_000, tolerance=1e-12)
    model.fit(X_scaled, y)
    assert model.score(X_scaled, y) > 0.999999
    assert model.intercept_ == pytest.approx(y.mean(), abs=1e-5)
    assert len(model.loss_history_) == model.n_iter_


def test_linear_regression_matches_sklearn_multivariate() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 3))
    y = X @ np.array([1.5, -2.0, 0.75]) + 4.0
    ours = LinearRegression(learning_rate=0.1, max_iter=10_000, tolerance=1e-14).fit(
        X, y
    )
    reference = SklearnLinearRegression().fit(X, y)
    np.testing.assert_allclose(ours.coef_, reference.coef_, atol=1e-5)
    assert ours.intercept_ == pytest.approx(reference.intercept_, abs=1e-5)


def test_linear_regression_can_disable_intercept() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = 2.0 * X[:, 0]
    model = LinearRegression(learning_rate=0.05, fit_intercept=False).fit(X, y)
    assert model.intercept_ == 0.0
    assert model.coef_[0] == pytest.approx(2.0, abs=1e-3)


def test_linear_regression_rejects_prediction_before_fit() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        LinearRegression().predict([[1.0]])


@pytest.mark.parametrize("learning_rate", [0, -1])
def test_linear_regression_rejects_bad_learning_rate(learning_rate: float) -> None:
    with pytest.raises(ValueError):
        LinearRegression(learning_rate=learning_rate)
