"""Command-line interface for reproducible demonstrations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .demos import DEMOS


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m ml_foundations",
        description="Run from-scratch machine learning demonstrations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo_parser = subparsers.add_parser("demo", help="Run an algorithm demo.")
    demo_parser.add_argument("algorithm", choices=sorted(DEMOS))
    demo_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory in which PNG plots are saved.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected demonstration."""
    arguments = build_parser().parse_args(argv)
    DEMOS[arguments.algorithm](arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
