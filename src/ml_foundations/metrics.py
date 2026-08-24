"""Evaluation metrics implemented with NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _paired_1d(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[NDArray, NDArray]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if true.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if true.shape[0] == 0:
        raise ValueError("Metrics require at least one sample.")
    return true, pred


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the average squared prediction error."""
    true, pred = _paired_1d(y_true, y_pred)
    try:
        residuals = true.astype(float) - pred.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Mean squared error requires numeric targets.") from exc
    if not np.all(np.isfinite(residuals)):
        raise ValueError("Targets must contain only finite values.")
    return float(np.mean(residuals**2))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the coefficient of determination."""
    true, pred = _paired_1d(y_true, y_pred)
    true_float = true.astype(float)
    denominator = float(np.sum((true_float - true_float.mean()) ** 2))
    numerator = float(np.sum((true_float - pred.astype(float)) ** 2))
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else 0.0
    return 1.0 - numerator / denominator


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the fraction of exactly matching labels."""
    true, pred = _paired_1d(y_true, y_pred)
    return float(np.mean(true == pred))


def confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike, *, labels: ArrayLike | None = None
) -> NDArray[np.int64]:
    """Count true labels by row and predicted labels by column."""
    true, pred = _paired_1d(y_true, y_pred)
    classes = (
        np.unique(np.concatenate((true, pred)))
        if labels is None
        else np.asarray(labels)
    )
    if classes.ndim != 1 or len(np.unique(classes)) != len(classes):
        raise ValueError("labels must be a 1D collection of unique values.")
    index = {label: position for position, label in enumerate(classes.tolist())}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    try:
        for actual, predicted in zip(true.tolist(), pred.tolist(), strict=True):
            matrix[index[actual], index[predicted]] += 1
    except KeyError as exc:
        raise ValueError("All observed labels must be included in labels.") from exc
    return matrix
