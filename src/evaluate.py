"""Evaluate the saved Iris model and record metrics for DVC and MLflow."""

import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "iris_model.joblib"
SPLIT_PATH = ROOT_DIR / "data" / "processed" / "iris_split.npz"
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"


def evaluate_model() -> dict[str, Any]:
    """Evaluate the model against the held-out 30 rows and save the results."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Saved train/test split not found: {SPLIT_PATH}")

    model = joblib.load(MODEL_PATH)
    with np.load(SPLIT_PATH, allow_pickle=False) as split:
        X_test = split["X_test"]
        y_test = split["y_test"]

    predictions = model.predict(X_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    report = classification_report(
        y_test,
        predictions,
        target_names=["setosa", "versicolor", "virginica"],
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} (weighted)")
    print(f"Recall:    {metrics['recall']:.4f} (weighted)")
    print(f"F1-score:  {metrics['f1_score']:.4f} (weighted)")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix (rows are actual; columns are predicted):")
    print(matrix)
    print(f"\nSaved metrics to {METRICS_PATH.relative_to(ROOT_DIR)}")

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    try:
        mlflow.set_experiment("Iris Flower Classification")
        with mlflow.start_run(run_name="random-forest-evaluation"):
            mlflow.log_metrics(
                {
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                }
            )
            mlflow.log_artifact(str(METRICS_PATH), artifact_path="evaluation")
    except Exception as error:
        print(f"Warning: MLflow logging failed: {error}")

    return metrics


def main() -> None:
    evaluate_model()


if __name__ == "__main__":
    main()
