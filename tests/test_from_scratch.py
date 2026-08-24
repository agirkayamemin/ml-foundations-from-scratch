from pathlib import Path


def test_core_implementations_do_not_import_sklearn() -> None:
    package = Path(__file__).parents[1] / "src" / "ml_foundations"
    core_modules = [
        "linear_regression.py",
        "knn.py",
        "kmeans.py",
        "pca.py",
        "preprocessing.py",
    ]
    for module in core_modules:
        source = (package / module).read_text(encoding="utf-8")
        assert "sklearn" not in source
