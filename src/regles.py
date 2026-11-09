"""
Règles métier MECHA, chargées depuis config/regles_metier.yaml.

Point d'accès unique aux seuils, niveaux de risque et règles d'alerte, utilisé
par la préparation des données, l'entraînement, l'API et le dashboard.
Le chemin du fichier peut être surchargé par la variable MECHA_REGLES.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml

BASE_DIR     = Path(__file__).resolve().parent.parent
CHEMIN_DEFAUT = BASE_DIR / "config" / "regles_metier.yaml"

NIVEAUX_ANOMALIE = ("anomalie", "suspect", "normal")


class ConfigurationInvalide(ValueError):
    """Le fichier de règles métier est incohérent."""


# ---------------------------------------------------------------------------
# Chargement et validation
# ---------------------------------------------------------------------------

def valider_regles(regles: dict) -> None:
    """Vérifie la cohérence du fichier de règles ; lève ConfigurationInvalide sinon."""
    for cle in ("capteurs", "qualite_donnees", "niveaux_risque_panne",
                "niveaux_risque_rul", "anomalie", "alertes", "frequences"):
        if cle not in regles:
            raise ConfigurationInvalide(f"Section manquante : {cle}")

    for capteur, bornes in regles["capteurs"].items():
        if bornes["min"] >= bornes["max"]:
            raise ConfigurationInvalide(f"Bornes incohérentes pour {capteur}")

    seuils = [n["seuil_min"] for n in regles["niveaux_risque_panne"]]
    if seuils != sorted(seuils, reverse=True) or len(set(seuils)) != len(seuils):
        raise ConfigurationInvalide("niveaux_risque_panne : seuils non strictement décroissants")
    if seuils[-1] != 0:
        raise ConfigurationInvalide("niveaux_risque_panne : le dernier niveau doit avoir seuil_min: 0")
    if any(not 0 <= s <= 1 for s in seuils):
        raise ConfigurationInvalide("niveaux_risque_panne : seuils hors de [0, 1]")

    rul = [n["rul_max_h"] for n in regles["niveaux_risque_rul"]]
    if rul[-1] is not None or any(r is None for r in rul[:-1]):
        raise ConfigurationInvalide("niveaux_risque_rul : seul le dernier niveau a rul_max_h: null")
    if rul[:-1] != sorted(rul[:-1]):
        raise ConfigurationInvalide("niveaux_risque_rul : rul_max_h non croissants")

    niveaux_panne = {n["niveau"] for n in regles["niveaux_risque_panne"]}
    for source, connus in (("panne24h", niveaux_panne), ("anomalie", set(NIVEAUX_ANOMALIE))):
        regle = regles["alertes"][source]
        inconnus = set(regle["niveaux"]) - connus
        if inconnus:
            raise ConfigurationInvalide(f"alertes.{source} : niveaux inconnus {sorted(inconnus)}")
        if int(regle["lectures_consecutives"]) < 1:
            raise ConfigurationInvalide(f"alertes.{source} : lectures_consecutives doit être >= 1")


@lru_cache(maxsize=4)
def _charger(chemin: str) -> dict:
    with open(chemin, encoding="utf-8") as f:
        regles = yaml.safe_load(f)
    valider_regles(regles)
    return regles


def charger_regles(chemin: str | Path | None = None) -> dict:
    """Retourne les règles métier (mises en cache après le premier chargement)."""
    chemin = chemin or os.getenv("MECHA_REGLES") or CHEMIN_DEFAUT
    return _charger(str(chemin))


# ---------------------------------------------------------------------------
# Niveaux de risque
# ---------------------------------------------------------------------------

def niveau_panne(probabilite: float, regles: dict | None = None) -> tuple[str, str]:
    """(niveau, action) pour une probabilité de panne dans les 24 h."""
    regles = regles or charger_regles()
    for n in regles["niveaux_risque_panne"]:
        if probabilite >= n["seuil_min"]:
            return n["niveau"], n["action"]
    dernier = regles["niveaux_risque_panne"][-1]
    return dernier["niveau"], dernier["action"]


def seuils_panne(regles: dict | None = None) -> list[tuple[str, float]]:
    """[(niveau, seuil_min), ...] du plus critique au plus faible."""
    regles = regles or charger_regles()
    return [(n["niveau"], float(n["seuil_min"])) for n in regles["niveaux_risque_panne"]]


def couleurs_niveaux(regles: dict | None = None) -> dict[str, str]:
    regles = regles or charger_regles()
    return {n["niveau"]: n["couleur"] for n in regles["niveaux_risque_panne"]}


def niveau_rul(rul_heures: float, regles: dict | None = None) -> tuple[str, str]:
    """(niveau, action) pour un temps restant avant défaillance."""
    regles = regles or charger_regles()
    for n in regles["niveaux_risque_rul"]:
        if n["rul_max_h"] is None or rul_heures < n["rul_max_h"]:
            return n["niveau"], n["action"]
    dernier = regles["niveaux_risque_rul"][-1]
    return dernier["niveau"], dernier["action"]


def score_anomalie_normalise(score_brut: float, regles: dict | None = None) -> float:
    """Score Isolation Forest (score_samples, négatif) ramené dans [0, 1], 1 = très anormal."""
    regles = regles or charger_regles()
    p = regles["anomalie"]["normalisation"]
    return float(max(0.0, min(1.0, (-score_brut - p["decalage"]) / p["echelle"])))


def niveau_anomalie(est_anomalie: bool, score: float, regles: dict | None = None) -> tuple[str, str]:
    """(niveau, action) : anomalie / suspect / normal."""
    regles = regles or charger_regles()
    cfg = regles["anomalie"]
    if est_anomalie and score > cfg["seuil_anomalie_forte"]:
        niveau = "anomalie"
    elif est_anomalie:
        niveau = "suspect"
    else:
        niveau = "normal"
    return niveau, cfg["actions"][niveau]


# ---------------------------------------------------------------------------
# Déclenchement des alertes
# ---------------------------------------------------------------------------

def lectures_requises(source: str, regles: dict | None = None) -> int:
    regles = regles or charger_regles()
    return int(regles["alertes"][source]["lectures_consecutives"])


def doit_alerter(source: str, niveaux_recents: list[str], regles: dict | None = None) -> bool:
    """
    True si les N dernières lectures (les plus récentes en premier) sont toutes
    dans un niveau déclencheur. `source` : « panne24h » ou « anomalie ».
    """
    regles = regles or charger_regles()
    regle = regles["alertes"][source]
    n = int(regle["lectures_consecutives"])
    recents = list(niveaux_recents)[:n]
    return len(recents) == n and all(niv in regle["niveaux"] for niv in recents)
