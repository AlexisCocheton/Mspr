"""
Module de préparation des données MECHA.

Charge le dataset synthétique unifié (reference/generate_dataset.py)
et le prépare pour l'entraînement ML.

Source  : data/raw/mecha_dataset_full.csv
Sortie  : data/processed/mecha_unified_prepared.csv
          data/processed/category_maps.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

BASE_DIR      = Path(__file__).resolve().parent.parent
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

SENSOR_COLS = [
    "temperature_C", "vibration_mm_s", "courant_A",
    "pression_bar",  "vitesse_tr_min",
]
WINDOW = 24  # fenêtre glissante horaire = 24 h


# ---------------------------------------------------------------------------
# 1. Chargement
# ---------------------------------------------------------------------------

def load_mecha_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Charge le dataset unifié MECHA."""
    path = raw_dir / "mecha_dataset_full.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}\n"
            "Copiez mecha_dataset_full.csv dans data/raw/  "
            "ou lancez : python reference/generate_dataset.py"
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute moyennes et écarts-types glissants sur 24 h par machine."""
    df = df.sort_values(["machine_id", "timestamp"]).copy()
    for col in SENSOR_COLS:
        prefix = col.split("_")[0]   # temperature, vibration, courant, pression, vitesse
        df[f"{prefix}_mean_24h"] = (
            df.groupby("machine_id")[col]
            .transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
        )
        df[f"{prefix}_std_24h"] = (
            df.groupby("machine_id")[col]
            .transform(lambda x: x.rolling(WINDOW, min_periods=1).std().fillna(0))
        )
    return df


def encode_categoricals(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Encode type_machine et usine_id, sauvegarde les mappings."""
    df = df.copy()
    le_type  = LabelEncoder()
    le_usine = LabelEncoder()
    df["type_machine_encoded"] = le_type.fit_transform(df["type_machine"])
    df["usine_encoded"]        = le_usine.fit_transform(df["usine_id"])

    maps = {
        "type_machine": {cls: int(i) for i, cls in enumerate(le_type.classes_)},
        "usine_id":     {cls: int(i) for i, cls in enumerate(le_usine.classes_)},
    }
    with open(output_dir / "category_maps.json", "w", encoding="utf-8") as f:
        json.dump(maps, f, indent=2, ensure_ascii=False)

    return df


# ---------------------------------------------------------------------------
# 3. Pipeline principale
# ---------------------------------------------------------------------------

def prepare_mecha(raw_dir: Path = RAW_DIR, output_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Charge, enrichit et sauvegarde le dataset unifié MECHA."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Chargement du dataset MECHA...")
    df = load_mecha_data(raw_dir)
    print(f"  {len(df):,} lignes | {df['machine_id'].nunique()} machines "
          f"| {df['usine_id'].nunique()} usines")

    print("Construction des features glissantes (24 h)...")
    df = build_rolling_features(df)

    print("Encodage des variables catégorielles...")
    df = encode_categoricals(df, output_dir)

    # Label prédictif : panne imminente dans les 24 h suivantes
    # (rul_heures <= 24, que la machine soit déjà en panne ou non)
    df["panne_dans_24h"] = (df["rul_heures"] <= 24).astype(int)

    out = output_dir / "mecha_unified_prepared.csv"
    df.to_csv(out, index=False)

    print(f"\nDataset prêt : {out}")
    print(f"  Shape              : {df.shape}")
    print(f"  Taux en_panne      : {df['en_panne'].mean():.2%}")
    print(f"  Taux panne_24h     : {df['panne_dans_24h'].mean():.2%}")
    print(f"  RUL moyen          : {df['rul_heures'].mean():.1f} h")
    print(f"  États machines     : {df['etat_machine'].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    prepare_mecha()
