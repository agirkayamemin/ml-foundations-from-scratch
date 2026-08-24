# Reproducible examples

The project examples are exposed through the command-line interface so that the
same tested implementation powers both terminal summaries and saved figures.

```bash
python -m ml_foundations demo linear-regression --output-dir outputs
python -m ml_foundations demo knn --output-dir outputs
python -m ml_foundations demo kmeans --output-dir outputs
python -m ml_foundations demo pca --output-dir outputs
```

All datasets and pseudo-random processes use fixed seeds.
