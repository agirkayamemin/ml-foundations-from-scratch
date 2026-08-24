import itertools

import numpy as np
import pytest
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from ml_foundations.kmeans import KMeans


def test_kmeans_finds_synthetic_clusters() -> None:
    X, expected = make_blobs(
        n_samples=120,
        centers=[(-6, -5), (0, 6), (6, -5)],
        cluster_std=0.45,
        random_state=5,
    )
    model = KMeans(n_clusters=3, random_state=5).fit(X)
    assert adjusted_rand_score(expected, model.labels_) > 0.95
    assert model.inertia_ >= 0
    assert model.n_iter_ <= model.max_iter


def test_kmeans_is_reproducible() -> None:
    X, _ = make_blobs(n_samples=40, centers=2, random_state=3)
    first = KMeans(n_clusters=2, random_state=11).fit(X)
    second = KMeans(n_clusters=2, random_state=11).fit(X)
    np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_)
    np.testing.assert_array_equal(first.labels_, second.labels_)


def test_kmeans_matches_reference_centers_ignoring_label_order() -> None:
    X = np.array([[-1.1], [-0.9], [9.9], [10.1]])
    ours = KMeans(n_clusters=2, random_state=0).fit(X)
    reference = SklearnKMeans(n_clusters=2, random_state=0, n_init=1).fit(X)
    ours_sorted = np.sort(ours.cluster_centers_[:, 0])
    reference_sorted = np.sort(reference.cluster_centers_[:, 0])
    np.testing.assert_allclose(ours_sorted, reference_sorted)


def test_kmeans_handles_empty_cluster_safely() -> None:
    X = np.array([[0.0], [0.0], [0.0], [10.0]])
    model = KMeans(n_clusters=3, random_state=1, max_iter=10).fit(X)
    assert np.all(np.isfinite(model.cluster_centers_))
    assert np.isfinite(model.inertia_)


def test_kmeans_fit_predict_equals_labels() -> None:
    X = np.array(list(itertools.product([0.0, 10.0], repeat=2)))
    model = KMeans(n_clusters=2, random_state=2)
    labels = model.fit_predict(X)
    np.testing.assert_array_equal(labels, model.labels_)


def test_kmeans_rejects_more_clusters_than_samples() -> None:
    with pytest.raises(ValueError, match="exceed"):
        KMeans(n_clusters=3).fit([[0], [1]])
