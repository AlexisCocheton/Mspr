"""Tests pour le module d'entraînement des modèles MECHA."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_training import prepare_features, FEATURE_COLS


@pytest.fixture
def sample_prepared_df():
    """Crée un DataFrame MECHA préparé de test (post data_preparation)."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "temperature_C": np.random.normal(82, 12, n),
        "vibration_mm_s": np.random.normal(2.8, 0.5, n),
        "courant_A": np.random.normal(20, 3, n),
        "pression_bar": np.random.normal(5.8, 0.4, n),
        "vitesse_tr_min": np.random.normal(1400, 100, n),
        "age_machine_h": np.random.randint(5000, 20000, n),
        "h_depuis_maintenance": np.random.randint(0, 800, n),
        "type_machine_encoded": np.random.randint(0, 4, n),
        "usine_encoded": np.random.randint(0, 5, n),
        "en_panne": np.random.choice([0, 1], n, p=[0.97, 0.03]),
        "panne_dans_24h": np.random.choice([0, 1], n, p=[0.97, 0.03]),
        "rul_heures": np.random.randint(1, 720, n),
    })


def test_feature_cols_count():
    """Vérifie que FEATURE_COLS contient 9 features."""
    assert len(FEATURE_COLS) == 9


def test_feature_cols_content():
    """Vérifie que les features attendues sont présentes."""
    expected = [
        "temperature_C", "vibration_mm_s", "courant_A",
        "pression_bar", "vitesse_tr_min",
        "age_machine_h", "h_depuis_maintenance",
        "type_machine_encoded", "usine_encoded",
    ]
    for col in expected:
        assert col in FEATURE_COLS


def test_prepare_features_shapes(sample_prepared_df):
    """Vérifie les dimensions de X et y."""
    X, y = prepare_features(sample_prepared_df, "en_panne")
    assert X.shape[0] == len(sample_prepared_df)
    assert X.shape[1] == len(FEATURE_COLS)
    assert len(y) == len(sample_prepared_df)


def test_prepare_features_no_nan(sample_prepared_df):
    """Vérifie qu'il n'y a pas de NaN dans X."""
    X, _ = prepare_features(sample_prepared_df, "en_panne")
    assert not np.isnan(X).any()


def test_prepare_features_binary_target_panne(sample_prepared_df):
    """Vérifie que la cible en_panne est binaire."""
    _, y = prepare_features(sample_prepared_df, "en_panne")
    assert set(y).issubset({0, 1})


def test_prepare_features_binary_target_24h(sample_prepared_df):
    """Vérifie que la cible panne_dans_24h est binaire."""
    _, y = prepare_features(sample_prepared_df, "panne_dans_24h")
    assert set(y).issubset({0, 1})


def test_prepare_features_rul_positive(sample_prepared_df):
    """Vérifie que les valeurs RUL sont positives."""
    _, y = prepare_features(sample_prepared_df, "rul_heures")
    assert (y >= 0).all()


def test_no_rolling_features_in_feature_cols():
    """Vérifie que les features glissantes 24h ne sont pas dans FEATURE_COLS."""
    for col in FEATURE_COLS:
        assert "mean_24h" not in col
        assert "std_24h" not in col
