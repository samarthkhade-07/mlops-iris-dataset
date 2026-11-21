"""FastAPI endpoints for health checks and Iris flower predictions."""

from math import isfinite

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.predict import predict_species

app = FastAPI(
    title="Iris Flower Classifier",
    description="Predict an Iris species from four measurements in centimeters.",
    version="1.0.0",
)


class FlowerMeasurements(BaseModel):
    """The four numeric Iris measurements accepted by the prediction API."""

    sepal_length: float = Field(gt=0, allow_inf_nan=False)
    sepal_width: float = Field(gt=0, allow_inf_nan=False)
    petal_length: float = Field(gt=0, allow_inf_nan=False)
    petal_width: float = Field(gt=0, allow_inf_nan=False)

    @field_validator(
        "sepal_length", "sepal_width", "petal_length", "petal_width"
    )
    @classmethod
    def measurement_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("Measurement must be a finite number.")
        return value


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report that the API process is responding."""
    return {"status": "healthy"}


@app.post("/predict")
def predict(flower: FlowerMeasurements) -> dict[str, str]:
    """Return the predicted species for the submitted flower measurements."""
    values = [
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width,
    ]
    try:
        species = predict_species(values)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"prediction": species}
