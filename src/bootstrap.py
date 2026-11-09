"""
Initialisation de la solution : prépare les données et entraîne les modèles
s'ils sont absents. Utilisé par le service « trainer » de docker-compose, pour
qu'un clone neuf du dépôt démarre sans étape manuelle.

Usage :
  python -m src.bootstrap           # n'entraîne que si des modèles manquent
  python -m src.bootstrap --force   # ré-entraîne tout
"""

import sys

from src.data_preparation import PROCESSED_DIR, prepare_mecha
from src.model_training import MODELS_DIR, train_all

MODELES_REQUIS = [
    "random_forest_classifier_mecha.joblib",
    "scaler_mecha.joblib",
    "random_forest_classifier_mecha_24h.joblib",
    "scaler_mecha_24h.joblib",
    "random_forest_regressor_mecha.joblib",
    "scaler_rul_mecha.joblib",
    "isolation_forest_mecha.joblib",
    "scaler_anomaly_mecha.joblib",
    "training_results.json",
]


def modeles_manquants() -> list[str]:
    return [f for f in MODELES_REQUIS if not (MODELS_DIR / f).exists()]


def main(force: bool = False) -> None:
    manquants = modeles_manquants()
    if not force and not manquants:
        print("Modèles présents — aucun entraînement nécessaire.")
        return

    print(f"Modèles manquants : {manquants or 'aucun (--force)'}")
    if force or not (PROCESSED_DIR / "mecha_unified_prepared.csv").exists():
        prepare_mecha()
    print("Entraînement des modèles (environ 10 à 15 minutes)...")
    train_all()

    if modeles_manquants():
        sys.exit(f"Entraînement incomplet, fichiers absents : {modeles_manquants()}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
