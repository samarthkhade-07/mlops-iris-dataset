"""Focused tests for Iris data, model training, predictions, and persistence."""

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris

from api.main import app
from src.data_loader import load_iris_dataset, split_dataset
from src.predict import SPECIES_NAMES, load_model, predict_species
from src.train import train_model

client = TestClient(app)


def test_dataset_has_expected_shape_and_species_distribution():
    iris = load_iris_dataset()

    assert iris.data.shape == (150, 4)
    assert iris.target.shape == (150,)
    assert iris.target_names.tolist() == ["setosa", "versicolor", "virginica"]
    assert np.bincount(iris.target).tolist() == [50, 50, 50]


def test_dataset_loader_uses_scikit_learn_builtin_data():
    iris = load_iris_dataset()

    assert np.array_equal(iris.data, load_iris().data)
    assert np.array_equal(iris.target, load_iris().target)


def test_split_keeps_all_rows_and_stratifies_each_class():
    X_train, X_test, y_train, y_test = split_dataset()

    assert X_train.shape == (120, 4)
    assert X_test.shape == (30, 4)
    assert np.bincount(y_train).tolist() == [40, 40, 40]
    assert np.bincount(y_test).tolist() == [10, 10, 10]


def test_model_trains_and_predicts_known_iris_species():
    X_train, X_test, y_train, _ = split_dataset()
    model = train_model(random_state=42)
    model.fit(X_train, y_train)

    assert model.predict(X_test).shape == (30,)
    prediction = predict_species([5.1, 3.5, 1.4, 0.2], model=model)
    assert prediction in {"Setosa", "Versicolor", "Virginica"}
    assert prediction == "Setosa"
    assert SPECIES_NAMES == ["Setosa", "Versicolor", "Virginica"]


def test_saved_model_can_be_loaded_and_used(tmp_path):
    iris = load_iris_dataset()
    model = train_model(random_state=42)
    model.fit(iris.data, iris.target)
    model_path = tmp_path / "iris_model.joblib"
    joblib.dump(model, model_path)

    loaded_model = load_model(model_path)

    assert predict_species([5.1, 3.5, 1.4, 0.2], model=loaded_model) == "Setosa"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction_endpoint(monkeypatch):
    monkeypatch.setattr("api.main.predict_species", lambda measurements: "Setosa")

    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": "Setosa"}


def test_prediction_endpoint_validates_measurements():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 0,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 422
