"""Train the Random Forest model and log the training run to MLflow."""

import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import display_dataset_info, split_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT_DIR / "params.yaml"
MODEL_PATH = ROOT_DIR / "models" / "iris_model.joblib"
SPLIT_PATH = ROOT_DIR / "data" / "processed" / "iris_split.npz"


def load_params() -> dict[str, Any]:
    """Read model and split settings from params.yaml."""
    with PARAMS_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_model(
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Create a Random Forest classifier with the requested settings."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    return model


def main() -> None:
    """Split the built-in dataset, train the model, and save its artifacts."""
    display_dataset_info()
    params = load_params()
    data_params = params["data"]
    model_params = params["model"]

    X_train, X_test, y_train, y_test = split_dataset(
        test_size=data_params["test_size"],
        random_state=data_params["random_state"],
    )
    print(f"\nTraining observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")

    model = train_model(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        min_samples_split=model_params["min_samples_split"],
        random_state=model_params["random_state"],
    )
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    np.savez_compressed(
        SPLIT_PATH, X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train
    )
    print(f"Saved trained model to {MODEL_PATH.relative_to(ROOT_DIR)}")

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    try:
        mlflow.set_experiment("Iris Flower Classification")
        with mlflow.start_run(run_name="random-forest-training"):
            mlflow.log_params(
                {
                    "model": "RandomForestClassifier",
                    "n_estimators": model_params["n_estimators"],
                    "max_depth": model_params["max_depth"],
                    "min_samples_split": model_params["min_samples_split"],
                    "random_state": model_params["random_state"],
                    "test_size": data_params["test_size"],
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                }
            )
            mlflow.sklearn.log_model(
                model,
                name="iris-random-forest",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            print("Training parameters and model logged to MLflow.")
    except Exception as error:
        print(f"Warning: MLflow logging failed: {error}")


if __name__ == "__main__":
    main()
