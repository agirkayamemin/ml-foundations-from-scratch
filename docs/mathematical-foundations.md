# Mathematical foundations

## Standardization

For feature `j`, each value is transformed as

```text
z_ij = (x_ij - mean_j) / standard_deviation_j
```

The implementation uses population variance (`ddof=0`). A constant feature has
zero variance; its scale is safely replaced by one, so its transformed values are
zero without division by zero.

## Linear Regression and gradient descent

Predictions combine feature weights and an intercept:

```text
y_hat = Xw + b
MSE = (1 / n) * sum((y_hat - y)^2)
```

The derivative of MSE points in the direction of greatest increase. Batch gradient
descent moves in the opposite direction:

```text
w = w - learning_rate * (2 / n) * X.T @ (y_hat - y)
b = b - learning_rate * 2 * mean(y_hat - y)
```

Training stops when consecutive losses differ by no more than the tolerance or the
maximum iteration count is reached.

## k-Nearest Neighbors

Euclidean distance between vectors `a` and `b` is

```text
distance(a, b) = sqrt(sum((a_j - b_j)^2))
```

The nearest `k` training labels vote on the prediction. Ties are resolved by the
smallest sorted class label, making predictions deterministic. Standardization is
important because otherwise a large-scale feature can dominate distance.

## K-Means

K-Means alternates between assigning each sample to its nearest centroid and
replacing each centroid with the mean of its assigned samples. Inertia is the sum
of squared distances to assigned centers. Multiple reproducible random starts are
evaluated and the lowest-inertia result is retained. An empty cluster is moved to
a currently farthest sample.

Cluster numbers carry no semantic order. Evaluation therefore uses Adjusted Rand
Index or silhouette score rather than direct equality with reference labels.

## Principal Component Analysis

PCA centers the data and forms the sample covariance matrix:

```text
C = X_centered.T @ X_centered / (n - 1)
```

`numpy.linalg.eigh` returns eigenvalues and orthonormal eigenvectors for this
symmetric matrix. Eigenvectors are sorted by decreasing eigenvalue; the leading
vectors define directions of greatest variance. Projection multiplies centered
data by these directions. Eigenvector signs are arbitrary, so comparisons use
absolute alignment and explained variance rather than exact signed values.
