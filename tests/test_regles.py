"""Tests du fichier de règles métier (config/regles_metier.yaml) et de son interprétation."""

import copy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import regles  # noqa: E402

REGLES = regles.charger_regles()


def test_fichier_de_regles_valide():
    regles.valider_regles(REGLES)


@pytest.mark.parametrize("proba, attendu", [
    (1.0, "critique"), (0.8, "critique"), (0.79, "élevé"),
    (0.5, "élevé"), (0.3, "moyen"), (0.29, "faible"), (0.0, "faible"),
])
def test_niveau_panne(proba, attendu):
    assert regles.niveau_panne(proba, REGLES)[0] == attendu


@pytest.mark.parametrize("rul, attendu", [
    (0, "critique"), (5.9, "critique"), (6, "élevé"),
    (23.9, "moyen"), (24, "faible"), (500, "faible"),
])
def test_niveau_rul(rul, attendu):
    assert regles.niveau_rul(rul, REGLES)[0] == attendu


def test_score_anomalie_borne_entre_0_et_1():
    assert regles.score_anomalie_normalise(-10.0, REGLES) == 1.0
    assert regles.score_anomalie_normalise(0.5, REGLES) == 0.0


def test_niveau_anomalie():
    seuil = REGLES["anomalie"]["seuil_anomalie_forte"]
    assert regles.niveau_anomalie(True, seuil + 0.01, REGLES)[0] == "anomalie"
    assert regles.niveau_anomalie(True, seuil - 0.01, REGLES)[0] == "suspect"
    assert regles.niveau_anomalie(False, 0.99, REGLES)[0] == "normal"


def test_doit_alerter_lectures_consecutives():
    n = regles.lectures_requises("panne24h", REGLES)
    niveau = REGLES["alertes"]["panne24h"]["niveaux"][0]
    assert regles.doit_alerter("panne24h", [niveau] * n, REGLES)
    assert not regles.doit_alerter("panne24h", [niveau] * (n - 1), REGLES)
    assert not regles.doit_alerter("panne24h", ["faible"] + [niveau] * n, REGLES)


def test_seuils_panne_ordonnes():
    seuils = [s for _, s in regles.seuils_panne(REGLES)]
    assert seuils == sorted(seuils, reverse=True) and seuils[-1] == 0


@pytest.mark.parametrize("modifier, message", [
    (lambda r: r["niveaux_risque_panne"].reverse(), "décroissants"),
    (lambda r: r["niveaux_risque_panne"][-1].update(seuil_min=0.1), "seuil_min: 0"),
    (lambda r: r["capteurs"]["temperature_C"].update(min=500), "Bornes"),
    (lambda r: r["alertes"]["panne24h"].update(niveaux=["inconnu"]), "inconnus"),
    (lambda r: r["alertes"]["panne24h"].update(lectures_consecutives=0), ">= 1"),
    (lambda r: r.pop("frequences"), "frequences"),
])
def test_configuration_invalide_rejetee(modifier, message):
    invalide = copy.deepcopy(REGLES)
    modifier(invalide)
    with pytest.raises(regles.ConfigurationInvalide, match=message):
        regles.valider_regles(invalide)


def test_chargement_d_un_autre_fichier(tmp_path):
    """Un autre jeu de règles (ex. par usine) se charge sans toucher au code."""
    autre = copy.deepcopy(REGLES)
    autre["alertes"]["panne24h"]["lectures_consecutives"] = 5
    chemin = tmp_path / "regles_site.yaml"
    chemin.write_text(yaml.safe_dump(autre, allow_unicode=True), encoding="utf-8")
    assert regles.lectures_requises("panne24h", regles.charger_regles(chemin)) == 5
