"""Load the saved model and turn a measurement row into a species name."""

from pathlib import Path
from typing import Sequence

import joblib
from sklearn.datasets import load_iris

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "iris_model.joblib"
SPECIES_NAMES = [name.title() for name in load_iris().target_names]


def load_model(model_path: Path = MODEL_PATH):
    """Load a model saved with joblib, reporting a clear error if absent."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run `python -m src.train` first."
        )
    return joblib.load(model_path)


def predict_species(measurements: Sequence[float], model=None) -> str:
    """Predict an Iris species for four measurements, in centimeters."""
    if len(measurements) != 4:
        raise ValueError("Exactly four flower measurements are required.")
    if model is None:
        model = load_model()
    class_id = int(model.predict([list(measurements)])[0])
    return SPECIES_NAMES[class_id]
