"""
Module de préparation des données MECHA.

Charge le dataset synthétique BRUT (reference/generate_dataset.py), le nettoie
et le prépare pour l'entraînement ML.

Source  : data/raw/mecha_dataset_full.csv           (brut, avec défauts capteurs)
Sortie  : data/processed/mecha_unified_prepared.csv (nettoyé + features)
          data/processed/category_maps.json
          data/processed/rapport_qualite.json       (bilan du nettoyage)

Les règles de nettoyage (bornes physiques, durée de capteur figé, durée
maximale d'imputation) proviennent de config/regles_metier.yaml.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

BASE_DIR      = Path(__file__).resolve().parent.parent
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

if str(BASE_DIR) not in sys.path:          # exécution directe : python src/data_preparation.py
    sys.path.insert(0, str(BASE_DIR))
from src.regles import charger_regles  # noqa: E402

SENSOR_COLS = [
    "temperature_C", "vibration_mm_s", "courant_A",
    "pression_bar",  "vitesse_tr_min",
]
WINDOW = 24  # fenêtre glissante horaire = 24 h


# ---------------------------------------------------------------------------
# 1. Chargement
# ---------------------------------------------------------------------------

def load_mecha_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Charge le dataset brut MECHA."""
    path = raw_dir / "mecha_dataset_full.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}\n"
            "Lancez : python reference/generate_dataset.py"
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


# ---------------------------------------------------------------------------
# 2. Nettoyage (qualité des données)
# ---------------------------------------------------------------------------

def _par_machine(df: pd.DataFrame, serie: pd.Series):
    return serie.groupby(df["machine_id"], sort=False)


def nettoyer_donnees(df: pd.DataFrame, regles: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Nettoie les mesures capteurs, dans l'ordre :
      1. suppression des doublons de transmission (même machine, même horodatage) ;
      2. valeurs hors bornes physiques -> valeur manquante ;
      3. capteur figé (même valeur pendant N heures ou plus) -> valeur manquante,
         sauf la première mesure de l'épisode ;
      4. trous de mesure de N heures au plus comblés par la dernière valeur connue
         (ce que l'atelier a aussi en temps réel) ;
      5. lignes encore incomplètes écartées.
    Retourne (df_nettoye, rapport) ; les labels ne sont pas modifiés.
    """
    regles  = regles or charger_regles()
    bornes  = regles["capteurs"]
    qualite = regles["qualite_donnees"]
    rapport = {"lignes_brutes": int(len(df))}

    df = df.sort_values(["machine_id", "timestamp"], kind="stable")
    avant = len(df)
    df = df.drop_duplicates(subset=["machine_id", "timestamp"], keep="first").reset_index(drop=True)
    rapport["doublons_supprimes"] = int(avant - len(df))
    rapport["valeurs_manquantes_brutes"] = int(df[SENSOR_COLS].isna().sum().sum())

    n = 0
    for col in SENSOR_COLS:
        b = bornes[col]
        masque = df[col].notna() & ((df[col] < b["min"]) | (df[col] > b["max"]))
        df.loc[masque, col] = np.nan
        n += int(masque.sum())
    rapport["valeurs_hors_bornes"] = n

    duree_fige = int(qualite["duree_min_capteur_fige_h"])
    n = 0
    for col in SENSOR_COLS:
        s = df[col]
        debut_episode = s.isna() | (s != _par_machine(df, s).shift())
        episode = debut_episode.cumsum()
        taille = episode.groupby(episode).transform("size")
        masque = (taille >= duree_fige) & ~debut_episode
        df.loc[masque, col] = np.nan
        n += int(masque.sum())
    rapport["valeurs_capteur_fige"] = n

    max_trou = int(qualite["imputation_max_h"])
    n = 0
    for col in SENSOR_COLS:
        s = df[col]
        manquant = s.isna()
        debut_trou = manquant & ~_par_machine(df, manquant).shift(fill_value=False).astype(bool)
        trou = debut_trou.cumsum().where(manquant)
        taille_trou = trou.map(trou.value_counts())
        derniere_valeur = _par_machine(df, s).ffill()
        masque = manquant & (taille_trou <= max_trou) & derniere_valeur.notna()
        df.loc[masque, col] = derniere_valeur[masque]
        n += int(masque.sum())
    rapport["valeurs_imputees"] = n

    avant = len(df)
    df = df.dropna(subset=SENSOR_COLS).reset_index(drop=True)
    rapport["lignes_ecartees"] = int(avant - len(df))
    rapport["lignes_nettoyees"] = int(len(df))
    return df, rapport


# ---------------------------------------------------------------------------
# 3. Feature engineering
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
# 4. Pipeline principale
# ---------------------------------------------------------------------------

def prepare_mecha(raw_dir: Path = RAW_DIR, output_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Charge, nettoie, enrichit et sauvegarde le dataset unifié MECHA."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Chargement du dataset MECHA (brut)...")
    df = load_mecha_data(raw_dir)
    print(f"  {len(df):,} lignes | {df['machine_id'].nunique()} machines "
          f"| {df['usine_id'].nunique()} usines")

    print("Nettoyage des données capteurs...")
    df, rapport = nettoyer_donnees(df)
    for k, v in rapport.items():
        print(f"  {k:<26} : {v:,}")
    with open(output_dir / "rapport_qualite.json", "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)

    print("Construction des features glissantes (24 h)...")
    df = build_rolling_features(df)

    print("Encodage des variables catégorielles...")
    df = encode_categoricals(df, output_dir)

    # Label prédictif : la prochaine panne commence dans les 24 h.
    # Pendant une panne, rul_heures pointe vers la panne SUIVANTE (>= 300 h),
    # donc les relevés en_panne = 1 ont panne_dans_24h = 0.
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
