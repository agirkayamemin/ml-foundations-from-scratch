"""Reproducible demonstrations for the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .kmeans import KMeans
from .knn import KNeighborsClassifier
from .linear_regression import LinearRegression
from .metrics import confusion_matrix
from .pca import PCA
from .preprocessing import StandardScaler, train_test_split


def _finish(summary: dict[str, Any], output_dir: Path | None) -> dict[str, Any]:
    if output_dir is not None:
        summary["output_dir"] = str(output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _plotter(output_dir: Path | None):
    if output_dir is None:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    return plt


def demo_linear_regression(output_dir: Path | None = None) -> dict[str, Any]:
    """Run a deterministic regression experiment and optional plots."""
    from sklearn.linear_model import LinearRegression as ReferenceLinearRegression

    rng = np.random.default_rng(42)
    X = np.linspace(-3, 3, 120).reshape(-1, 1)
    y = 2.5 * X[:, 0] - 1.0 + rng.normal(0, 0.35, len(X))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression(learning_rate=0.1, max_iter=5_000).fit(
        X_train_scaled, y_train
    )
    reference = ReferenceLinearRegression().fit(X_train_scaled, y_train)
    summary = {
        "algorithm": "linear-regression",
        "iterations": model.n_iter_,
        "r2": round(model.score(X_test_scaled, y_test), 4),
        "reference_r2": round(reference.score(X_test_scaled, y_test), 4),
    }
    plt = _plotter(output_dir)
    if plt is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(model.loss_history_)
        plt.xlabel("Iteration")
        plt.ylabel("Mean squared error")
        plt.title("Linear Regression Loss")
        plt.tight_layout()
        plt.savefig(output_dir / "linear-regression-loss.png", dpi=150)
        plt.close()
        order = np.argsort(X_test[:, 0])
        plt.figure(figsize=(6, 4))
        plt.scatter(X_test[:, 0], y_test, label="Observed", alpha=0.75)
        plt.plot(
            X_test[order, 0],
            model.predict(X_test_scaled)[order],
            color="tab:red",
            label="Predicted",
        )
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.title("Linear Regression Predictions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "linear-regression-predictions.png", dpi=150)
        plt.close()
    return _finish(summary, output_dir)


def demo_knn(output_dir: Path | None = None) -> dict[str, Any]:
    """Evaluate KNN on Iris and compare several k values."""
    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier as ReferenceKNN

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    scores = {}
    for k in (1, 3, 5, 7, 9):
        scores[k] = (
            KNeighborsClassifier(n_neighbors=k)
            .fit(X_train, y_train)
            .score(X_test, y_test)
        )
    model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    predictions = model.predict(X_test)
    reference = ReferenceKNN(n_neighbors=5).fit(X_train, y_train)
    summary = {
        "algorithm": "knn",
        "accuracy": round(model.score(X_test, y_test), 4),
        "reference_accuracy": round(reference.score(X_test, y_test), 4),
        "scores_by_k": {str(k): round(value, 4) for k, value in scores.items()},
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    plt = _plotter(output_dir)
    if plt is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(list(scores), list(scores.values()), marker="o")
        plt.xlabel("k")
        plt.ylabel("Accuracy")
        plt.title("KNN Accuracy by k")
        plt.xticks(list(scores))
        plt.tight_layout()
        plt.savefig(output_dir / "knn-k-comparison.png", dpi=150)
        plt.close()
        matrix = np.asarray(summary["confusion_matrix"])
        plt.figure(figsize=(5, 4))
        plt.imshow(matrix, cmap="Blues")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                plt.text(column, row, matrix[row, column], ha="center", va="center")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("KNN Confusion Matrix")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output_dir / "knn-confusion-matrix.png", dpi=150)
        plt.close()
    return _finish(summary, output_dir)


def demo_kmeans(output_dir: Path | None = None) -> dict[str, Any]:
    """Cluster synthetic and Iris data and report label-independent quality."""
    from sklearn.cluster import KMeans as ReferenceKMeans
    from sklearn.datasets import load_iris, make_blobs
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    X, expected = make_blobs(
        n_samples=180,
        centers=[(-6, -5), (0, 6), (6, -5)],
        cluster_std=0.8,
        random_state=42,
    )
    model = KMeans(n_clusters=3, random_state=42).fit(X)
    reference = ReferenceKMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
    iris_X, iris_y = load_iris(return_X_y=True)
    iris_scaled = StandardScaler().fit_transform(iris_X)
    iris_model = KMeans(n_clusters=3, random_state=42).fit(iris_scaled)
    summary = {
        "algorithm": "kmeans",
        "adjusted_rand_index": round(adjusted_rand_score(expected, model.labels_), 4),
        "inertia": round(model.inertia_, 4),
        "reference_inertia": round(reference.inertia_, 4),
        "iris_adjusted_rand_index": round(
            adjusted_rand_score(iris_y, iris_model.labels_), 4
        ),
        "iris_silhouette": round(silhouette_score(iris_scaled, iris_model.labels_), 4),
    }
    plt = _plotter(output_dir)
    if plt is not None:
        plt.figure(figsize=(6, 4))
        plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis", s=24)
        plt.scatter(
            model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            c="red",
            marker="X",
            s=150,
            label="Centroids",
        )
        plt.title("K-Means on Synthetic Data")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "kmeans-synthetic.png", dpi=150)
        plt.close()
        iris_2d = PCA(n_components=2).fit_transform(iris_scaled)
        plt.figure(figsize=(6, 4))
        plt.scatter(iris_2d[:, 0], iris_2d[:, 1], c=iris_model.labels_, cmap="viridis")
        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        plt.title("K-Means on Iris (PCA projection)")
        plt.tight_layout()
        plt.savefig(output_dir / "kmeans-iris.png", dpi=150)
        plt.close()
    return _finish(summary, output_dir)


def demo_pca(output_dir: Path | None = None) -> dict[str, Any]:
    """Project Iris to two dimensions and compare explained variance."""
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA as ReferencePCA

    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    model = PCA(n_components=2).fit(X)
    transformed = model.transform(X)
    reference = ReferencePCA(n_components=2).fit(X)
    summary = {
        "algorithm": "pca",
        "explained_variance_ratio": np.round(
            model.explained_variance_ratio_, 4
        ).tolist(),
        "reference_explained_variance_ratio": np.round(
            reference.explained_variance_ratio_, 4
        ).tolist(),
        "total_explained_variance": round(
            float(model.explained_variance_ratio_.sum()), 4
        ),
    }
    plt = _plotter(output_dir)
    if plt is not None:
        plt.figure(figsize=(6, 4))
        for label in np.unique(y):
            mask = y == label
            plt.scatter(transformed[mask, 0], transformed[mask, 1], label=str(label))
        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        plt.title("PCA Projection of Iris")
        plt.legend(title="Class")
        plt.tight_layout()
        plt.savefig(output_dir / "pca-iris.png", dpi=150)
        plt.close()
    return _finish(summary, output_dir)


DEMOS = {
    "linear-regression": demo_linear_regression,
    "knn": demo_knn,
    "kmeans": demo_kmeans,
    "pca": demo_pca,
}
