import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

from ml_foundations.knn import KNeighborsClassifier
from ml_foundations.preprocessing import StandardScaler, train_test_split


def test_knn_predicts_simple_clusters() -> None:
    X = [[0, 0], [0, 1], [5, 5], [5, 6]]
    y = ["left", "left", "right", "right"]
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    np.testing.assert_array_equal(
        model.predict([[0.1, 0.2], [5.1, 5.2]]), ["left", "right"]
    )


def test_knn_resolves_vote_tie_by_sorted_class() -> None:
    model = KNeighborsClassifier(n_neighbors=2).fit([[-1], [1]], [2, 1])
    assert model.predict([[0]])[0] == 1


def test_knn_matches_sklearn_on_iris() -> None:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    ours = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    reference = SklearnKNN(n_neighbors=5).fit(X_train, y_train)
    assert ours.score(X_test, y_test) >= 0.9
    np.testing.assert_array_equal(ours.predict(X_test), reference.predict(X_test))


def test_knn_rejects_too_many_neighbors() -> None:
    with pytest.raises(ValueError, match="exceed"):
        KNeighborsClassifier(n_neighbors=3).fit([[0], [1]], [0, 1])


def test_knn_rejects_prediction_before_fit() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        KNeighborsClassifier().predict([[0]])
