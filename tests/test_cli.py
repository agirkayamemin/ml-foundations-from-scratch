from pathlib import Path

import pytest

from ml_foundations.__main__ import main


@pytest.mark.parametrize(
    ("algorithm", "expected_file"),
    [
        ("linear-regression", "linear-regression-loss.png"),
        ("knn", "knn-confusion-matrix.png"),
        ("kmeans", "kmeans-synthetic.png"),
        ("pca", "pca-iris.png"),
    ],
)
def test_demo_cli_generates_plots(
    algorithm: str, expected_file: str, tmp_path: Path
) -> None:
    assert main(["demo", algorithm, "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / expected_file).stat().st_size > 0
