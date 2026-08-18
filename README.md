# ML Foundations from Scratch

A learning-focused Python project that implements fundamental machine learning algorithms from scratch using NumPy. The project aims to connect the mathematical foundations of machine learning with tested, reusable, and well-documented Python code.

## What “from scratch” means

The core algorithms are implemented without using ready-made machine learning models from libraries such as scikit-learn. NumPy is used for numerical computation, while scikit-learn may be used to load datasets, calculate reference metrics, and compare results.

## Planned algorithms

- Linear Regression with gradient descent
- k-Nearest Neighbors classification
- K-Means clustering
- Principal Component Analysis

## Development setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install the project and its development dependencies:

```bash
python -m pip install --editable ".[dev]"
```

## Tests and code quality

Run the automated tests:

```bash
python -m pytest
```

Run the code quality checks:

```bash
python -m ruff check .
```

## Project status

This project is in early development and is intended for educational and portfolio purposes. It is not a production-ready machine learning framework.
