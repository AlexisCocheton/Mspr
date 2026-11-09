"""
Simulateur de capteurs IoT MECHA.

Rejoue les lignes BRUTES de mecha_dataset_full.csv vers le webhook
/webhook/sensor de l'API, en simulant les 30 machines IoT en production :
  - chaque relevé est envoyé avec son horodatage de mesure (measured_at) ;
  - un relevé incomplet (perte de mesure) n'est pas transmis ;
  - un relevé hors bornes physiques est transmis tel quel : l'API le rejette
    (HTTP 422), comme le ferait un contrôle de plausibilité à l'ingestion.

Variables d'environnement :
  API_URL             (défaut : http://mecha-api:8000)
  DATA_PATH           (défaut : data/raw/mecha_dataset_full.csv)
  SIMULATOR_DELAY     (défaut : frequences.simulateur_intervalle_s du fichier de règles)
  SIMULATOR_MAX_RETRY (défaut : 30 tentatives de connexion API au démarrage)
"""

import os
import sys
import time
import logging
from pathlib import Path

import httpx
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:          # exécution directe : python src/sensor_simulator.py
    sys.path.insert(0, str(BASE_DIR))
from src.regles import charger_regles  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SIMULATOR] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

API_URL    = os.getenv("API_URL",   "http://mecha-api:8000")
DATA_PATH  = os.getenv("DATA_PATH", str(BASE_DIR / "data" / "raw" / "mecha_dataset_full.csv"))
DELAY      = float(os.getenv("SIMULATOR_DELAY")
                   or charger_regles()["frequences"]["simulateur_intervalle_s"])
MAX_RETRY  = int(os.getenv("SIMULATOR_MAX_RETRY", "30"))

SENSOR_COLS = ["temperature_C", "vibration_mm_s", "courant_A", "pression_bar", "vitesse_tr_min"]

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


def build_payload(row: pd.Series) -> dict:
    return {
        "machine_id":           str(row["machine_id"]),
        "usine_id":             str(row["usine_id"]),
        "type_machine":         str(row["type_machine"]).strip(),
        "measured_at":          pd.Timestamp(row["timestamp"]).isoformat(),
        "temperature_C":        float(row["temperature_C"]),
        "vibration_mm_s":       float(row["vibration_mm_s"]),
        "courant_A":            float(row["courant_A"]),
        "pression_bar":         float(row["pression_bar"]),
        "vitesse_tr_min":       float(row["vitesse_tr_min"]),
        "age_machine_h":        int(row["age_machine_h"]),
        "h_depuis_maintenance": int(row["h_depuis_maintenance"]),
    }


def send_row(client: httpx.Client, row: pd.Series) -> str:
    """
    Envoie une ligne CSV vers POST /webhook/sensor.
    Retourne « ok », « rejete » (contrôle de plausibilité de l'API) ou « erreur ».
    """
    try:
        r = client.post(f"{API_URL}/webhook/sensor", json=build_payload(row), timeout=10.0)
    except httpx.RequestError as exc:
        logger.warning("  %s — Erreur réseau : %s", row["machine_id"], exc)
        return "erreur"
    if r.status_code == 422:
        logger.info("  %s — relevé rejeté (hors bornes physiques)", row["machine_id"])
        return "rejete"
    if r.status_code != 200:
        logger.warning("  %s — HTTP %d : %s", row["machine_id"], r.status_code, r.text[:120])
        return "erreur"
    for alerte in r.json().get("alertes", []):
        logger.warning("  ALERTE %s — %s : %s",
                       alerte["niveau"].upper(), alerte["machine_id"], alerte["message"])
    return "ok"


def load_data(path: str) -> pd.DataFrame:
    """Charge le CSV brut (sans nettoyage : c'est le rôle de l'API et de la préparation)."""
    logger.info("Chargement du dataset : %s", path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info("  %d lignes | %d timestamps uniques | %d machines",
                len(df), df["timestamp"].nunique(), df["machine_id"].nunique())

    before = len(df)
    df = df[df["type_machine"].str.strip().isin(VALID_TYPES) & df["usine_id"].isin(VALID_USINES)]
    if len(df) < before:
        logger.warning("  %d lignes supprimées (type/usine inconnu)", before - len(df))
    return df


def run_simulation(df: pd.DataFrame) -> None:
    """Boucle de simulation infinie sur les timestamps du CSV."""
    groupes    = [g for _, g in df.groupby("timestamp", sort=True)]
    total      = len(groupes)
    loop_count = 0

    with httpx.Client() as client:
        while True:
            loop_count += 1
            logger.info("=== Boucle %d — %d timestamps à envoyer ===", loop_count, total)

            for i, group in enumerate(groupes, 1):
                complets = group.dropna(subset=SENSOR_COLS)
                bilan = {"ok": 0, "rejete": 0, "erreur": 0}
                for _, row in complets.iterrows():
                    bilan[send_row(client, row)] += 1
                logger.info(
                    "[%d/%d] ts=%s  machines=%d  ok=%d  rejetés=%d  non_transmis=%d  erreurs=%d",
                    i, total, group["timestamp"].iloc[0], len(group), bilan["ok"],
                    bilan["rejete"], len(group) - len(complets), bilan["erreur"],
                )
                time.sleep(DELAY)

            logger.info("=== Fin boucle %d — reprise depuis le début ===", loop_count)


def main() -> None:
    logger.info("Simulateur MECHA démarré")
    logger.info("  API_URL          : %s", API_URL)
    logger.info("  DATA_PATH        : %s", DATA_PATH)
    logger.info("  SIMULATOR_DELAY  : %.1f s", DELAY)

    wait_for_api(API_URL, MAX_RETRY)

    df = load_data(DATA_PATH)
    run_simulation(df)


if __name__ == "__main__":
    main()
