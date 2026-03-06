"""Tests pour l'API FastAPI."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_check():
    """Vérifie que l'endpoint /health répond."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_model_info():
    """Vérifie que l'endpoint /model/info répond."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data


def test_predict_ai4i_valid_input():
    """Teste la prédiction AI4I avec des données valides."""
    payload = {
        "air_temp_c": 25.0,
        "process_temp_c": 35.5,
        "rotational_speed_rpm": 1500,
        "torque_nm": 42.0,
        "tool_wear_min": 100,
        "quality_type": "M",
    }
    response = client.post("/predict/ai4i", json=payload)
    # 200 si le modèle est chargé, 503 sinon
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1
        assert data["risk_level"] in ["faible", "moyen", "élevé", "critique"]


def test_predict_pdm_valid_input():
    """Teste la prédiction PdM avec des données valides."""
    payload = {
        "voltage": 170.0,
        "rotation_speed": 450.0,
        "pressure": 100.0,
        "vibration": 40.0,
        "machine_age_years": 10,
        "machine_model": "model3",
    }
    response = client.post("/predict/pdm", json=payload)
    assert response.status_code in [200, 503]


def test_predict_rul_valid_input():
    """Teste la prédiction RUL avec des données valides."""
    payload = {
        "voltage": 170.0,
        "rotation_speed": 450.0,
        "pressure": 100.0,
        "vibration": 40.0,
        "machine_age_years": 10,
        "machine_model": "model3",
    }
    response = client.post("/predict/rul", json=payload)
    assert response.status_code in [200, 503]


def test_predict_ai4i_missing_field():
    """Teste que l'API rejette les données incomplètes."""
    payload = {
        "air_temp_c": 25.0,
        # Champs manquants
    }
    response = client.post("/predict/ai4i", json=payload)
    assert response.status_code == 422  # Validation error
