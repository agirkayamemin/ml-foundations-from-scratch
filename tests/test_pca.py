import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

from ml_foundations.pca import PCA


def test_pca_reduces_dimensions_and_centers_projection() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 4))
    transformed = PCA(n_components=2).fit_transform(X)
    assert transformed.shape == (50, 2)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)


def test_pca_matches_sklearn_variance_despite_sign_ambiguity() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(80, 3)) @ np.array([[2, 0, 0], [0, 1, 0], [0, 0, 0.2]])
    ours = PCA(n_components=2).fit(X)
    reference = SklearnPCA(n_components=2).fit(X)
    np.testing.assert_allclose(ours.explained_variance_, reference.explained_variance_)
    np.testing.assert_allclose(
        ours.explained_variance_ratio_, reference.explained_variance_ratio_
    )
    alignment = np.abs(np.sum(ours.components_ * reference.components_, axis=1))
    np.testing.assert_allclose(alignment, 1.0, atol=1e-10)


def test_pca_full_inverse_reconstructs_input() -> None:
    X = np.array([[1.0, 2.0], [3.0, 1.0], [4.0, 5.0]])
    model = PCA().fit(X)
    np.testing.assert_allclose(
        model.inverse_transform(model.transform(X)), X, atol=1e-12
    )


def test_pca_constant_data_has_zero_variance_ratio() -> None:
    model = PCA(n_components=1).fit(np.ones((4, 2)))
    np.testing.assert_array_equal(model.explained_variance_ratio_, [0.0])


def test_pca_rejects_too_many_components() -> None:
    with pytest.raises(ValueError, match="exceed"):
        PCA(n_components=3).fit([[1, 2], [3, 4]])


def test_pca_rejects_transform_before_fit() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        PCA(n_components=1).transform([[1, 2]])
