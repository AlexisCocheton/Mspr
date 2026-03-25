"""
Simulateur de capteurs IoT MECHA.

Rejoue les lignes de mecha_dataset_full.csv vers le webhook /webhook/sensor
de l'API, en simulant le comportement de 30 machines IoT en production.

Variables d'environnement :
  API_URL            (défaut : http://mecha-api:8000)
  DATA_PATH          (défaut : /app/data/raw/mecha_dataset_full.csv)
  SIMULATOR_DELAY    (défaut : 2.0 s entre chaque timestamp simulé)
  SIMULATOR_MAX_RETRY (défaut : 30 tentatives de connexion API au démarrage)
"""

import os
import sys
import time
import logging
from pathlib import Path

import httpx
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SIMULATOR] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

API_URL    = os.getenv("API_URL",   "http://mecha-api:8000")
DATA_PATH  = os.getenv("DATA_PATH", str(Path(__file__).resolve().parent.parent
                                        / "data" / "raw" / "mecha_dataset_full.csv"))
DELAY      = float(os.getenv("SIMULATOR_DELAY",     "2.0"))
MAX_RETRY  = int(os.getenv("SIMULATOR_MAX_RETRY",   "30"))

# Types de machines valides (contrainte Pydantic de l'API)
VALID_TYPES = {"CNC-Fraisage", "CNC-Tournage", "Découpe-Laser", "Centre-Usinage"}
VALID_USINES = {"USN-FR-01", "USN-FR-02", "USN-FR-03", "USN-ES-01", "USN-ES-02"}


def wait_for_api(base_url: str, max_attempts: int) -> None:
    """Bloque jusqu'à ce que GET /health réponde 200."""
    for attempt in range(1, max_attempts + 1):
        try:
            r = httpx.get(f"{base_url}/health", timeout=3.0)
            if r.status_code == 200:
                logger.info("API disponible après %d tentative(s).", attempt)
                return
        except httpx.RequestError:
            pass
        logger.info("Attente de l'API... tentative %d/%d", attempt, max_attempts)
        time.sleep(2)
    logger.error("L'API n'est pas disponible après %d tentatives. Arrêt.", max_attempts)
    sys.exit(1)


def send_row(client: httpx.Client, row: pd.Series) -> bool:
    """
    Envoie une ligne CSV vers POST /webhook/sensor.
    Retourne True si succès, False sinon.
    """
    payload = {
        "machine_id":           str(row["machine_id"]),
        "usine_id":             str(row["usine_id"]),
        "type_machine":         str(row["type_machine"]).strip(),
        "temperature_C":        float(row["temperature_C"]),
        "vibration_mm_s":       float(row["vibration_mm_s"]),
        "courant_A":            float(row["courant_A"]),
        "pression_bar":         float(row["pression_bar"]),
        "vitesse_tr_min":       float(row["vitesse_tr_min"]),
        "age_machine_h":        int(row["age_machine_h"]),
        "h_depuis_maintenance": int(row["h_depuis_maintenance"]),
    }
    try:
        r = client.post(f"{API_URL}/webhook/sensor", json=payload, timeout=10.0)
        if r.status_code != 200:
            logger.warning(
                "  %s — HTTP %d : %s",
                row["machine_id"], r.status_code, r.text[:120],
            )
            return False
        return True
    except httpx.RequestError as exc:
        logger.warning("  %s — Erreur réseau : %s", row["machine_id"], exc)
        return False


def load_and_clean(path: str) -> pd.DataFrame:
    """Charge le CSV et nettoie les valeurs hors bornes Pydantic."""
    logger.info("Chargement du dataset : %s", path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info("  %d lignes | %d timestamps uniques | %d machines",
                len(df), df["timestamp"].nunique(), df["machine_id"].nunique())

    # Filtrer les types/usines inconnus (ne doivent pas arriver avec ce dataset)
    before = len(df)
    df = df[df["type_machine"].str.strip().isin(VALID_TYPES)]
    df = df[df["usine_id"].isin(VALID_USINES)]
    if len(df) < before:
        logger.warning("  %d lignes supprimées (type/usine inconnu)", before - len(df))

    # Clamp des valeurs hors bornes de validation Pydantic
    df["temperature_C"]  = df["temperature_C"].clip(upper=299.0)
    df["vibration_mm_s"] = df["vibration_mm_s"].clip(upper=19.99)
    df["courant_A"]      = df["courant_A"].clip(lower=0.0, upper=99.9)
    df["pression_bar"]   = df["pression_bar"].clip(lower=0.0, upper=19.9)
    df["vitesse_tr_min"] = df["vitesse_tr_min"].clip(lower=0.0, upper=2999.0)
    df["age_machine_h"]  = df["age_machine_h"].clip(lower=0, upper=199999)
    df["h_depuis_maintenance"] = df["h_depuis_maintenance"].clip(lower=0, upper=9999)

    return df


def run_simulation(df: pd.DataFrame) -> None:
    """Boucle de simulation infinie sur les timestamps du CSV."""
    timestamps = sorted(df["timestamp"].unique())
    total      = len(timestamps)
    loop_count = 0

    with httpx.Client() as client:
        while True:
            loop_count += 1
            logger.info("=== Boucle %d — %d timestamps à envoyer ===", loop_count, total)

            for i, ts in enumerate(timestamps, 1):
                group = df[df["timestamp"] == ts]
                ok    = sum(send_row(client, row) for _, row in group.iterrows())
                logger.info(
                    "[%d/%d] ts=%s  machines=%d  envois_ok=%d",
                    i, total, ts, len(group), ok,
                )
                time.sleep(DELAY)

            logger.info("=== Fin boucle %d — reprise depuis le début ===", loop_count)


def main() -> None:
    logger.info("Simulateur MECHA démarré")
    logger.info("  API_URL          : %s", API_URL)
    logger.info("  DATA_PATH        : %s", DATA_PATH)
    logger.info("  SIMULATOR_DELAY  : %.1f s", DELAY)

    wait_for_api(API_URL, MAX_RETRY)

    df = load_and_clean(DATA_PATH)
    run_simulation(df)


if __name__ == "__main__":
    main()
