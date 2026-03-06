"""Tests pour le module d'entraînement des modèles."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_training import (
    prepare_ai4i_for_training,
    FEATURE_COLS_AI4I,
)


@pytest.fixture
def sample_prepared_ai4i():
    """Crée un DataFrame AI4I déjà nettoyé (post data_preparation)."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "air_temp_c": np.random.normal(25, 2, n),
        "process_temp_c": np.random.normal(35, 2, n),
        "rotational_speed_rpm": np.random.normal(1500, 200, n),
        "torque_nm": np.random.normal(40, 10, n),
        "tool_wear_min": np.random.randint(0, 250, n),
        "temp_diff": np.random.normal(10, 1, n),
        "power_w": np.random.normal(6000, 1000, n),
        "torque_speed_ratio": np.random.normal(0.027, 0.005, n),
        "quality_type": np.random.choice(["L", "M", "H"], n),
        "machine_failure": np.random.choice([0, 1], n, p=[0.97, 0.03]),
    })


def test_prepare_ai4i_shapes(sample_prepared_ai4i):
    """Vérifie les dimensions de X et y."""
    X, y, feat_names = prepare_ai4i_for_training(sample_prepared_ai4i)
    assert X.shape[0] == len(sample_prepared_ai4i)
    assert X.shape[1] == len(feat_names)
    assert len(y) == len(sample_prepared_ai4i)


def test_prepare_ai4i_feature_names(sample_prepared_ai4i):
    """Vérifie que les noms de features sont corrects."""
    _, _, feat_names = prepare_ai4i_for_training(sample_prepared_ai4i)
    for col in FEATURE_COLS_AI4I:
        assert col in feat_names
    assert "quality_encoded" in feat_names


def test_prepare_ai4i_no_nan(sample_prepared_ai4i):
    """Vérifie qu'il n'y a pas de NaN dans X."""
    X, _, _ = prepare_ai4i_for_training(sample_prepared_ai4i)
    assert not np.isnan(X).any()


def test_prepare_ai4i_binary_target(sample_prepared_ai4i):
    """Vérifie que y est binaire."""
    _, y, _ = prepare_ai4i_for_training(sample_prepared_ai4i)
    unique_values = set(y)
    assert unique_values.issubset({0, 1})
