"""Machine learning algorithms implemented from scratch with NumPy."""

from .kmeans import KMeans
from .knn import KNeighborsClassifier
from .linear_regression import LinearRegression
from .metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from .pca import PCA
from .preprocessing import StandardScaler, train_test_split

__all__ = [
    "PCA",
    "KMeans",
    "KNeighborsClassifier",
    "LinearRegression",
    "StandardScaler",
    "accuracy_score",
    "confusion_matrix",
    "mean_squared_error",
    "r2_score",
    "train_test_split",
]
