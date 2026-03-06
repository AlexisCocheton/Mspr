"""Tests pour le module de préparation des données."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_preparation import clean_ai4i, load_ai4i_data, RAW_DIR


@pytest.fixture
def sample_ai4i_df():
    """Crée un DataFrame AI4I de test."""
    return pd.DataFrame({
        "UDI": [1, 2, 3],
        "Product ID": ["M14860", "L47181", "H29424"],
        "Type": ["M", "L", "H"],
        "Air temperature [K]": [298.1, 298.2, 298.4],
        "Process temperature [K]": [308.6, 308.7, 308.9],
        "Rotational speed [rpm]": [1551, 1408, 1782],
        "Torque [Nm]": [42.8, 46.3, 23.9],
        "Tool wear [min]": [0, 3, 24],
        "Machine failure": [0, 0, 0],
        "TWF": [0, 0, 0],
        "HDF": [0, 0, 0],
        "PWF": [0, 0, 0],
        "OSF": [0, 0, 0],
        "RNF": [0, 0, 0],
    })


def test_clean_ai4i_renames_columns(sample_ai4i_df):
    """Vérifie que les colonnes sont renommées correctement."""
    result = clean_ai4i(sample_ai4i_df)
    assert "air_temp_c" in result.columns
    assert "process_temp_c" in result.columns
    assert "rotational_speed_rpm" in result.columns
    assert "machine_failure" in result.columns


def test_clean_ai4i_temperature_conversion(sample_ai4i_df):
    """Vérifie la conversion K -> °C."""
    result = clean_ai4i(sample_ai4i_df)
    expected = 298.1 - 273.15
    assert abs(result["air_temp_c"].iloc[0] - expected) < 0.01


def test_clean_ai4i_derived_features(sample_ai4i_df):
    """Vérifie la création des features dérivées."""
    result = clean_ai4i(sample_ai4i_df)
    assert "temp_diff" in result.columns
    assert "power_w" in result.columns
    assert "torque_speed_ratio" in result.columns

    # temp_diff = process - air
    expected_diff = (308.6 - 273.15) - (298.1 - 273.15)
    assert abs(result["temp_diff"].iloc[0] - expected_diff) < 0.01


def test_clean_ai4i_power_calculation(sample_ai4i_df):
    """Vérifie le calcul de la puissance."""
    result = clean_ai4i(sample_ai4i_df)
    expected_power = 42.8 * 1551 * 2 * np.pi / 60
    assert abs(result["power_w"].iloc[0] - expected_power) < 0.1


def test_clean_ai4i_no_data_loss(sample_ai4i_df):
    """Vérifie qu'aucune ligne n'est perdue."""
    result = clean_ai4i(sample_ai4i_df)
    assert len(result) == len(sample_ai4i_df)


def test_load_ai4i_data_shape():
    """Vérifie que le dataset AI4I se charge correctement."""
    if (RAW_DIR / "ai4i2020.csv").exists():
        df = load_ai4i_data()
        assert len(df) == 10000
        assert "Machine failure" in df.columns
