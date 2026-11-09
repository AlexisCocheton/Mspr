"""Tests pour le module d'entraînement des modèles MECHA."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_training import (
    prepare_features, FEATURE_COLS,
    balance_dataset, compute_metrics, alert_breakdown,
)


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


def test_balance_dataset_50_50():
    """Le rééquilibrage produit autant de pannes que de non-pannes."""
    X = np.arange(200).reshape(100, 2)
    y = np.array([1] * 10 + [0] * 90)
    Xb, yb = balance_dataset(X, y)
    assert len(yb) == 20
    assert yb.sum() == 10
    assert Xb.shape == (20, 2)


def test_modele_naif_paradoxe_accuracy():
    """
    Un modèle qui répond toujours « pas de panne » obtient ~97 % d'accuracy
    sur des données réelles (2,3 % de pannes) sans détecter aucune panne.
    C'est pourquoi les modèles sont aussi évalués sur un test 50/50.
    """
    y_reel = np.array([1] * 23 + [0] * 977)
    m_reel = compute_metrics(y_reel, np.zeros_like(y_reel), np.zeros(len(y_reel)))
    assert m_reel["accuracy"] == pytest.approx(0.977)
    assert m_reel["recall"] == 0.0
    assert m_reel["f1_score"] == 0.0

    y_5050 = np.array([1] * 50 + [0] * 50)
    m_5050 = compute_metrics(y_5050, np.zeros_like(y_5050), np.zeros(len(y_5050)))
    assert m_5050["accuracy"] == pytest.approx(0.5)


def test_alert_breakdown_par_niveau():
    """Comptage des alertes et des fausses alertes par niveau de risque."""
    y_true  = np.array([1,   0,    0,   1,   0,   1])
    y_proba = np.array([0.9, 0.85, 0.6, 0.4, 0.1, 0.2])
    out = alert_breakdown(y_true, y_proba)

    critique = out["niveaux"]["critique"]
    assert critique["n_alertes"] == 2
    assert critique["vraies_pannes"] == 1
    assert critique["taux_fausses_alertes"] == pytest.approx(0.5)

    assert out["niveaux"]["élevé"]["n_alertes"] == 1
    assert out["niveaux"]["élevé"]["taux_fausses_alertes"] == pytest.approx(1.0)
    assert out["niveaux"]["moyen"]["vraies_pannes"] == 1
    assert out["pannes_non_alertees"] == 1


def test_alert_breakdown_seuils_inclusifs():
    """Une probabilité exactement égale au seuil tombe dans le niveau supérieur (comme l'API)."""
    out = alert_breakdown(np.array([1, 1, 1]), np.array([0.8, 0.5, 0.3]))
    assert out["niveaux"]["critique"]["n_alertes"] == 1
    assert out["niveaux"]["élevé"]["n_alertes"] == 1
    assert out["niveaux"]["moyen"]["n_alertes"] == 1


def test_alert_breakdown_niveau_vide():
    """Aucune alerte dans un niveau : taux de fausses alertes non défini."""
    out = alert_breakdown(np.array([0, 1]), np.array([0.1, 0.2]))
    assert out["niveaux"]["critique"]["taux_fausses_alertes"] is None
    assert out["pannes_non_alertees"] == 1


# ---------------------------------------------------------------------------
# Validation de la règle d'alerte (rejeu temporel)
# ---------------------------------------------------------------------------

from src.model_training import (  # noqa: E402
    debuts_de_panne, simuler_alertes, evaluer_alertes_evenements, alertes_suivies_de_panne,
)


def test_debuts_de_panne():
    assert list(debuts_de_panne(np.array([0, 1, 1, 0, 0, 1]))) == [1, 5]
    assert list(debuts_de_panne(np.array([1, 1, 0]))) == [0]


def test_simuler_alertes_lectures_consecutives():
    heures = np.arange(10.0)
    proba  = np.array([.9, .1, .9, .9, .9, .1, .9, .9, .1, .1])
    alertes = simuler_alertes(proba, np.zeros(10), heures, seuil=0.8, lectures=2, delai_h=0)
    assert list(alertes) == [3, 4, 7]


def test_simuler_alertes_delai_entre_alertes():
    heures = np.arange(10.0)
    alertes = simuler_alertes(np.full(10, 0.9), np.zeros(10), heures, seuil=0.8, lectures=1, delai_h=5)
    assert list(alertes) == [0, 5]


def test_simuler_alertes_ignore_machine_deja_en_panne():
    heures = np.arange(4.0)
    alertes = simuler_alertes(np.full(4, 0.9), np.array([1, 1, 1, 1]), heures, 0.8, 1, 0)
    assert len(alertes) == 0


def test_evaluer_alertes_evenements():
    alertes = np.array([10.0, 50.0])
    pannes  = np.array([20.0, 200.0])
    r = evaluer_alertes_evenements(alertes, pannes, horizon_h=24)
    assert r["alertes_justifiees"] == 1      # 10 -> panne à 20 ; 50 -> rien avant 74
    assert r["pannes_anticipees"] == 1       # la panne à 200 n'a pas d'alerte dans [176, 200)
    assert r["anticipations_h"] == [10.0]


def test_alertes_suivies_de_panne_exclut_les_alertes_non_jugeables():
    alertes = np.array([10.0, 95.0])
    jugeables, suivies = alertes_suivies_de_panne(alertes, np.array([30.0]), fin_h=100, horizon_h=24)
    assert (jugeables, suivies) == (1, 1)    # l'alerte à 95 h n'a pas 24 h de recul
