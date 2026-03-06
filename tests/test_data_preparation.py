"""Tests pour le module de préparation des données MECHA."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_preparation import (
    load_mecha_data,
    build_rolling_features,
    encode_categoricals,
    SENSOR_COLS,
    WINDOW,
)


@pytest.fixture
def sample_mecha_df():
    """Crée un DataFrame MECHA minimal pour les tests."""
    np.random.seed(42)
    n = 100
    machines = ["MCH-001"] * 50 + ["MCH-002"] * 50
    timestamps = pd.date_range("2024-01-01", periods=50, freq="h").tolist() * 2
    return pd.DataFrame({
        "timestamp": timestamps,
        "machine_id": machines,
        "usine_id": ["USN-FR-01"] * n,
        "type_machine": ["CNC-Fraisage"] * n,
        "temperature_C": np.random.normal(82, 10, n),
        "vibration_mm_s": np.random.normal(2.8, 0.4, n),
        "courant_A": np.random.normal(20, 3, n),
        "pression_bar": np.random.normal(5.8, 0.3, n),
        "vitesse_tr_min": np.random.normal(1400, 80, n),
        "age_machine_h": np.random.randint(5000, 20000, n),
        "h_depuis_maintenance": np.random.randint(0, 800, n),
        "score_degradation": np.random.uniform(0, 1, n),
        "rul_heures": np.random.randint(1, 720, n),
        "en_panne": np.random.choice([0, 1], n, p=[0.97, 0.03]),
        "etat_machine": np.random.choice(["normal", "a_risque", "critique", "en_panne"], n),
    })


def test_build_rolling_features_adds_columns(sample_mecha_df):
    """Vérifie que les colonnes glissantes sont ajoutées."""
    result = build_rolling_features(sample_mecha_df)
    for col in SENSOR_COLS:
        prefix = col.split("_")[0]
        assert f"{prefix}_mean_24h" in result.columns
        assert f"{prefix}_std_24h" in result.columns


def test_build_rolling_features_no_data_loss(sample_mecha_df):
    """Vérifie qu'aucune ligne n'est perdue."""
    result = build_rolling_features(sample_mecha_df)
    assert len(result) == len(sample_mecha_df)


def test_build_rolling_features_no_nan_in_mean(sample_mecha_df):
    """Vérifie que les moyennes glissantes ne contiennent pas de NaN."""
    result = build_rolling_features(sample_mecha_df)
    for col in SENSOR_COLS:
        prefix = col.split("_")[0]
        assert result[f"{prefix}_mean_24h"].isna().sum() == 0


def test_encode_categoricals_creates_columns(sample_mecha_df):
    """Vérifie que les colonnes encodées sont créées."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = encode_categoricals(sample_mecha_df, Path(tmpdir))
    assert "type_machine_encoded" in result.columns
    assert "usine_encoded" in result.columns


def test_encode_categoricals_saves_json(sample_mecha_df):
    """Vérifie que le fichier category_maps.json est créé."""
    with tempfile.TemporaryDirectory() as tmpdir:
        encode_categoricals(sample_mecha_df, Path(tmpdir))
        maps_path = Path(tmpdir) / "category_maps.json"
        assert maps_path.exists()
        with open(maps_path) as f:
            maps = json.load(f)
        assert "type_machine" in maps
        assert "usine_id" in maps


def test_encode_categoricals_integer_values(sample_mecha_df):
    """Vérifie que les encodages sont des entiers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = encode_categoricals(sample_mecha_df, Path(tmpdir))
    assert result["type_machine_encoded"].dtype in [int, np.int64, np.int32]
    assert result["usine_encoded"].dtype in [int, np.int64, np.int32]


def test_load_mecha_data_file_not_found():
    """Vérifie que FileNotFoundError est levé si le fichier est absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            load_mecha_data(Path(tmpdir))


def test_window_constant():
    """Vérifie que la fenêtre glissante est bien 24h."""
    assert WINDOW == 24
