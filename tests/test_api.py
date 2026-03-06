"""Tests pour l'API FastAPI MECHA."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

VALID_PAYLOAD = {
    "temperature_C": 85.0,
    "vibration_mm_s": 3.1,
    "courant_A": 22.0,
    "pression_bar": 5.5,
    "vitesse_tr_min": 1350,
    "age_machine_h": 12000,
    "h_depuis_maintenance": 400,
    "type_machine": "CNC-Fraisage",
    "usine_id": "USN-FR-01",
}


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


def test_predict_panne_valid_input():
    """Teste la prédiction en_panne avec des données valides."""
    response = client.post("/predict/panne", json=VALID_PAYLOAD)
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0
        assert data["risk_level"] in ["faible", "moyen", "élevé", "critique"]
        assert "recommendation" in data


def test_predict_panne24h_valid_input():
    """Teste la prédiction panne_dans_24h avec des données valides."""
    response = client.post("/predict/panne24h", json=VALID_PAYLOAD)
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0
        assert data["risk_level"] in ["faible", "moyen", "élevé", "critique"]


def test_predict_rul_valid_input():
    """Teste la prédiction RUL avec des données valides."""
    response = client.post("/predict/rul", json=VALID_PAYLOAD)
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "estimated_rul_hours" in data
        assert data["estimated_rul_hours"] >= 0
        assert data["risk_level"] in ["faible", "moyen", "élevé", "critique"]
        assert "recommendation" in data


def test_predict_panne_missing_field():
    """Teste que l'API rejette les données incomplètes."""
    payload = {"temperature_C": 85.0}
    response = client.post("/predict/panne", json=payload)
    assert response.status_code == 422


def test_predict_panne24h_missing_field():
    """Teste que l'API rejette les données incomplètes pour panne24h."""
    payload = {"vibration_mm_s": 3.1}
    response = client.post("/predict/panne24h", json=payload)
    assert response.status_code == 422


def test_predict_rul_missing_field():
    """Teste que l'API rejette les données incomplètes pour RUL."""
    payload = {"pression_bar": 5.5}
    response = client.post("/predict/rul", json=payload)
    assert response.status_code == 422
