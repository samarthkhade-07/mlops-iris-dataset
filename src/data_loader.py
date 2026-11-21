"""Load and inspect the built-in Iris dataset."""

from typing import Any

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_iris_dataset() -> Any:
    """Load the original Iris dataset bundled with scikit-learn."""
    return load_iris()


def display_dataset_info() -> None:
    """Print the dataset description, sample rows, and class distribution."""
    iris = load_iris_dataset()
    print(iris.DESCR)
    print("\nFirst five observations:")
    print(", ".join(iris.feature_names + ["target"]))
    for row, target in zip(iris.data[:5], iris.target[:5]):
        print(", ".join(f"{value:.1f}" for value in row), int(target))

    print(f"\nRows: {iris.data.shape[0]}")
    print(f"Columns (features): {iris.data.shape[1]}")
    print(f"Feature names: {iris.feature_names}")
    print(f"Target species: {list(iris.target_names)}")
    print(f"Missing values: {int(np.isnan(iris.data).sum())}")
    print("Observations per species:")
    for class_id, count in enumerate(np.bincount(iris.target)):
        print(f"  {iris.target_names[class_id]}: {int(count)}")


def split_dataset(
    test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a reproducible, stratified train/test split of all 150 rows."""
    iris = load_iris_dataset()
    return train_test_split(
        iris.data,
        iris.target,
        test_size=test_size,
        random_state=random_state,
        stratify=iris.target,
    )
