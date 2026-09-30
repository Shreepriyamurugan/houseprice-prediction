"""Integration & unit tests for the FastAPI prediction API."""

import json
import sys

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from lifinity.config import get_project_root

ROOT = get_project_root()
MODEL_PATH = ROOT / "models" / "model.joblib"
DEFAULTS_PATH = ROOT / "models" / "input_defaults.json"
SAMPLE_PATH = ROOT / "api" / "sample_request.json"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/model.joblib is missing; skipping API integration tests",
)


@pytest.fixture
def client():
    """TestClient fixture executing app lifespan."""
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model_type" in data
    assert "model_source" in data


def test_predict_sample(client):
    assert SAMPLE_PATH.exists()
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        sample = json.load(f)

    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()

    price = data["predicted_price"]
    low = data["price_range_low"]
    high = data["price_range_high"]

    assert 50000.0 <= price <= 800000.0
    assert low < price < high
    assert "model_type" in data
    assert "model_version" in data


def test_predict_minimal_required_fields(client):
    minimal = {
        "OverallQual": 7,
        "GrLivArea": 1500.0,
        "Neighborhood": "NAmes",
    }
    response = client.post("/predict", json=minimal)
    assert response.status_code == 200
    data = response.json()
    assert data["fields_defaulted"] > 70


def test_predict_invalid_overall_qual(client):
    payload = {
        "OverallQual": 15,
        "GrLivArea": 1500.0,
        "Neighborhood": "NAmes",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_neighborhood(client):
    payload = {
        "OverallQual": 7,
        "GrLivArea": 1500.0,
        "Neighborhood": "Atlantis",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_future_year_built_warning(client):
    payload = {
        "OverallQual": 7,
        "GrLivArea": 1500.0,
        "Neighborhood": "NAmes",
        "YearBuilt": 2020,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["warnings"]) > 0
    assert any("2010" in w for w in data["warnings"])


def test_predict_alias_1st_flr_sf(client):
    payload = {
        "OverallQual": 7,
        "GrLivArea": 1500.0,
        "Neighborhood": "NAmes",
        "1stFlrSF": 1000.0,
        "2ndFlrSF": 500.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price"] > 0


def test_predict_batch(client):
    batch = [
        {"OverallQual": 6, "GrLivArea": 1200.0, "Neighborhood": "NAmes"},
        {"OverallQual": 7, "GrLivArea": 1710.0, "Neighborhood": "CollgCr"},
        {"OverallQual": 8, "GrLivArea": 2200.0, "Neighborhood": "NridgHt"},
    ]
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 3


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "final_model_type" in data
    assert "test_metrics" in data
    tm = data["test_metrics"]
    assert "rmse_log" in tm
    assert "mae" in tm
    assert "mape" in tm
    assert "r2_log" in tm


def test_prediction_no_distortion_vs_direct_model(client):
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        sample = json.load(f)

    # API prediction
    api_response = client.post("/predict", json=sample)
    assert api_response.status_code == 200
    api_price = api_response.json()["predicted_price"]

    # Direct model evaluation on the exact filled dataframe row
    with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
        defaults = json.load(f)

    filled = dict(defaults)
    filled.update(sample)
    df_row = pd.DataFrame([filled])[list(defaults.keys())]

    model = joblib.load(MODEL_PATH)
    direct_pred = float(model.predict(df_row)[0])

    assert abs(api_price - direct_pred) <= 1.0
