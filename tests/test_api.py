"""
Tests pour l'API FastAPI MECHA.

Les modèles sont remplacés par de petits modèles entraînés à la volée et la
base PostgreSQL par une base en mémoire : les tests vérifient réellement les
prédictions, les règles d'alerte et l'acquittement, sans dépendre des
artefacts d'entraînement ni d'un serveur PostgreSQL (CI).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from src import api, regles  # noqa: E402

VALID_PAYLOAD = {
    "temperature_C": 85.0,
    "vibration_mm_s": 3.1,
    "courant_A": 22.0,
    "pression_bar": 5.5,
    "vitesse_tr_min": 1350,
    "age_machine_h": 12000,
    "h_depuis_maintenance": 400,
    "type_machine": "CNC-Fraisage",
    "usine_id": "USN-FR-01",
}
NIVEAUX_PANNE = {n for n, _ in regles.seuils_panne()}
NIVEAUX_RUL = {n["niveau"] for n in regles.charger_regles()["niveaux_risque_rul"]}


# ---------------------------------------------------------------------------
# Doubles de test
# ---------------------------------------------------------------------------

class ProbaFixe:
    """Classifieur qui renvoie toujours la même probabilité de panne."""
    feature_importances_ = np.full(9, 1 / 9)

    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        return np.tile([1 - self.p, self.p], (len(X), 1))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class AnomalieFixe:
    """Isolation Forest factice : verdict et score brut constants."""

    def __init__(self, est_anomalie: bool, score_brut: float = -0.4):
        self.est_anomalie, self.score_brut = est_anomalie, score_brut

    def predict(self, X):
        return np.full(len(X), -1 if self.est_anomalie else 1)

    def score_samples(self, X):
        return np.full(len(X), self.score_brut)


class FakeDB:
    """Base en mémoire exposant les fonctions de src.database utilisées par l'API."""

    def __init__(self):
        self.readings, self.predictions, self.alerts = [], [], []

    def init_schema(self):
        pass

    def is_db_available(self):
        return True

    def insert_reading(self, data):
        self.readings.append(data)
        return len(self.readings)

    def insert_prediction(self, reading_id, machine_id, usine_id, type_machine, panne24h, rul, anomaly):
        self.predictions.append({
            "id": len(self.predictions) + 1, "machine_id": machine_id, "usine_id": usine_id,
            "type_machine": type_machine, "predicted_at": datetime.now(timezone.utc),
            "panne24h_probability": panne24h["probability"],
            "panne24h_risk_level": panne24h["risk_level"],
            "anomaly_risk_level": anomaly["risk_level"],
        })
        return len(self.predictions)

    def get_recent_levels(self, machine_id, source, n):
        col = {"panne24h": "panne24h_risk_level", "anomalie": "anomaly_risk_level"}[source]
        rows = [p for p in self.predictions if p["machine_id"] == machine_id]
        return [p[col] for p in reversed(rows)][:n]

    def has_open_alert(self, machine_id, source):
        return any(a["machine_id"] == machine_id and a["source"] == source and a["statut"] == "ouverte"
                   for a in self.alerts)

    def insert_alert(self, data):
        alerte = {**data, "id": len(self.alerts) + 1, "statut": "ouverte", "notifiee": False,
                  "created_at": datetime.now(timezone.utc), "acquittee_at": None}
        self.alerts.append(alerte)
        return dict(alerte)

    def mark_alert_notified(self, alert_id):
        self.alerts[alert_id - 1]["notifiee"] = True

    def get_alerts(self, statut=None, limit=50):
        rows = [dict(a) for a in reversed(self.alerts) if statut in (None, a["statut"])]
        return rows[:limit]

    def acknowledge_alert(self, alert_id):
        if not 1 <= alert_id <= len(self.alerts):
            return None
        self.alerts[alert_id - 1].update(statut="acquittee", acquittee_at=datetime.now(timezone.utc))
        return dict(self.alerts[alert_id - 1])

    def get_recent_predictions(self, limit=50):
        return [dict(p) for p in reversed(self.predictions)][:limit]

    def get_last_prediction_per_machine(self):
        dernier = {}
        for p in self.predictions:
            dernier[p["machine_id"]] = dict(p)
        return list(dernier.values())


def petits_modeles() -> dict:
    """Petits modèles entraînés sur des relevés réalistes autour de VALID_PAYLOAD."""
    rng = np.random.default_rng(0)
    base = np.array([85.0, 3.1, 22.0, 5.5, 1350, 12000, 400, 0, 0])
    X = base * (1 + rng.normal(0, 0.1, size=(300, 9)))
    y = (X[:, 0] > base[0]).astype(int)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    return {
        "clf_panne": RandomForestClassifier(n_estimators=10, random_state=0).fit(Xs, y),
        "scaler_panne": scaler,
        "clf_24h": RandomForestClassifier(n_estimators=10, random_state=0).fit(Xs, y),
        "scaler_24h": scaler,
        "reg_rul": RandomForestRegressor(n_estimators=10, random_state=0).fit(Xs, X[:, 6]),
        "scaler_rul": scaler,
        "iso_forest": IsolationForest(n_estimators=20, random_state=0).fit(Xs),
        "scaler_anomaly": scaler,
        "category_maps": {"type_machine": {"CNC-Fraisage": 0}, "usine_id": {"USN-FR-01": 0}},
    }


@pytest.fixture
def fake_db(monkeypatch):
    base = FakeDB()
    monkeypatch.setattr(api, "db", base)
    monkeypatch.setattr(api, "DB_AVAILABLE", True)
    monkeypatch.delenv("MECHA_ALERT_WEBHOOK_URL", raising=False)
    return base


@pytest.fixture
def client(fake_db):
    with TestClient(api.app) as c:      # exécute le lifespan (chargement des règles et modèles)
        api.models.clear()
        api.models.update(petits_modeles())
        yield c
    api.models.clear()


def envoyer(client, machine="USN-FR-01-M01", **extra):
    return client.post("/webhook/sensor", json={**VALID_PAYLOAD, "machine_id": machine, **extra})


@pytest.fixture
def sans_anomalie(client):
    """Isole la règle panne 24 h : l'Isolation Forest ne signale rien."""
    api.models["iso_forest"] = AnomalieFixe(False)


# ---------------------------------------------------------------------------
# État et configuration
# ---------------------------------------------------------------------------

def test_health_check(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "clf_24h" in data["models_loaded"]


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    assert "available_models" in response.json()


def test_config_regles(client):
    data = client.get("/config/regles").json()
    assert {"capteurs", "niveaux_risque_panne", "alertes", "frequences"} <= data.keys()


# ---------------------------------------------------------------------------
# Prédictions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("endpoint", ["/predict/panne", "/predict/panne24h"])
def test_predict_classification(client, endpoint):
    response = client.post(endpoint, json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_level"] in NIVEAUX_PANNE
    assert data["recommendation"]


def test_predict_niveau_suit_les_regles(client):
    """Le niveau renvoyé est celui du fichier de règles pour la probabilité calculée."""
    api.models["clf_24h"] = ProbaFixe(0.85)
    data = client.post("/predict/panne24h", json=VALID_PAYLOAD).json()
    assert data["probability"] == pytest.approx(0.85)
    assert (data["risk_level"], data["recommendation"]) == regles.niveau_panne(0.85)


def test_predict_rul(client):
    data = client.post("/predict/rul", json=VALID_PAYLOAD).json()
    assert data["estimated_rul_hours"] >= 0
    assert data["risk_level"] in NIVEAUX_RUL


def test_predict_anomalie(client):
    data = client.post("/predict/anomalie", json=VALID_PAYLOAD).json()
    assert isinstance(data["is_anomaly"], bool)
    assert 0.0 <= data["anomaly_score"] <= 1.0
    assert data["risk_level"] in {"normal", "suspect", "anomalie"}


@pytest.mark.parametrize("endpoint", ["/predict/panne/explain", "/predict/panne24h/explain"])
def test_explain(client, endpoint):
    response = client.post(endpoint, json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert len(data["top_features"]) == 3
    assert data["methode"] in {"shap", "importance_globale"}
    assert {f["direction"] for f in data["top_features"]} <= {"hausse_risque", "baisse_risque"}


def test_predict_batch(client):
    response = client.post("/predict/batch", json={"lignes": [VALID_PAYLOAD, VALID_PAYLOAD]})
    assert response.status_code == 200
    resultats = response.json()["resultats"]
    assert len(resultats) == 2
    assert resultats[0]["rul"]["estimated_rul_hours"] >= 0


def test_predict_batch_limite(client):
    response = client.post("/predict/batch", json={"lignes": [VALID_PAYLOAD] * 1001})
    assert response.status_code == 422


def test_modeles_absents_503(client):
    api.models.clear()
    assert client.post("/predict/panne24h", json=VALID_PAYLOAD).status_code == 503


# ---------------------------------------------------------------------------
# Validation des entrées
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("endpoint, payload", [
    ("/predict/panne",    {"temperature_C": 85.0}),
    ("/predict/panne24h", {"vibration_mm_s": 3.1}),
    ("/predict/rul",      {"pression_bar": 5.5}),
])
def test_champs_manquants(client, endpoint, payload):
    assert client.post(endpoint, json=payload).status_code == 422


def test_valeur_hors_bornes_physiques(client):
    """Les bornes du fichier de règles servent de contrôle de plausibilité."""
    borne = regles.charger_regles()["capteurs"]["temperature_C"]["max"]
    payload = {**VALID_PAYLOAD, "temperature_C": borne + 1}
    assert client.post("/predict/panne24h", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# Webhook, alertes, acquittement
# ---------------------------------------------------------------------------

def test_webhook_persiste_lecture_et_prediction(client, fake_db):
    response = envoyer(client, measured_at="2025-06-01T08:00:00")
    assert response.status_code == 200
    data = response.json()
    assert data["machine_id"] == "USN-FR-01-M01"
    assert len(fake_db.readings) == 1 and len(fake_db.predictions) == 1
    assert fake_db.readings[0]["measured_at"].year == 2025


def test_alerte_apres_lectures_consecutives(client, fake_db, sans_anomalie):
    api.models["clf_24h"] = ProbaFixe(0.95)
    n = regles.lectures_requises("panne24h")
    for _ in range(n - 1):
        assert envoyer(client).json()["alertes"] == []
    alertes = envoyer(client).json()["alertes"]
    assert [a["source"] for a in alertes] == ["panne24h"]
    assert alertes[0]["niveau"] == regles.niveau_panne(0.95)[0]


def test_pas_d_alerte_pour_un_pic_isole(client, fake_db, sans_anomalie):
    n = regles.lectures_requises("panne24h")
    for i in range(2 * n):
        api.models["clf_24h"] = ProbaFixe(0.95 if i % 2 else 0.05)
        assert envoyer(client).json()["alertes"] == []
    assert fake_db.alerts == []


def test_pas_de_doublon_tant_que_l_alerte_est_ouverte(client, fake_db, sans_anomalie):
    api.models["clf_24h"] = ProbaFixe(0.95)
    n = regles.lectures_requises("panne24h")
    for _ in range(3 * n):
        envoyer(client)
    assert len([a for a in fake_db.alerts if a["source"] == "panne24h"]) == 1

    # Une fois acquittée, l'alerte peut se redéclencher
    assert client.post("/alertes/1/acquitter").json()["statut"] == "acquittee"
    envoyer(client)
    assert len([a for a in fake_db.alerts if a["source"] == "panne24h"]) == 2


def test_alertes_separees_par_machine(client, fake_db, sans_anomalie):
    api.models["clf_24h"] = ProbaFixe(0.95)
    for _ in range(regles.lectures_requises("panne24h")):
        envoyer(client, machine="USN-FR-01-M01")
        envoyer(client, machine="USN-FR-01-M02")
    machines = {a["machine_id"] for a in fake_db.alerts if a["source"] == "panne24h"}
    assert machines == {"USN-FR-01-M01", "USN-FR-01-M02"}


def test_notification_webhook_externe(client, fake_db, sans_anomalie, monkeypatch):
    envois = []

    class Reponse:
        def raise_for_status(self):
            pass

    monkeypatch.setenv("MECHA_ALERT_WEBHOOK_URL", "http://ticketing.local/alertes")
    monkeypatch.setattr(api.httpx, "post", lambda url, json, timeout: envois.append((url, json)) or Reponse())
    api.models["clf_24h"] = ProbaFixe(0.95)
    for _ in range(regles.lectures_requises("panne24h")):
        reponse = envoyer(client).json()
    assert reponse["alertes"][0]["notifiee"] is True
    assert envois[0][0] == "http://ticketing.local/alertes"
    assert envois[0][1]["machine_id"] == "USN-FR-01-M01"


def test_alerte_anomalie(client, fake_db):
    """Comportement capteur très atypique répété : alerte de source « anomalie »."""
    api.models["clf_24h"] = ProbaFixe(0.05)
    api.models["iso_forest"] = AnomalieFixe(True, score_brut=-1.0)
    for _ in range(regles.lectures_requises("anomalie")):
        alertes = envoyer(client).json()["alertes"]
    assert [(a["source"], a["niveau"]) for a in alertes] == [("anomalie", "anomalie")]


def test_lister_et_acquitter_alertes(client, fake_db, sans_anomalie):
    api.models["clf_24h"] = ProbaFixe(0.95)
    for _ in range(regles.lectures_requises("panne24h")):
        envoyer(client)
    ouvertes = client.get("/alertes", params={"statut": "ouverte"}).json()
    assert ouvertes["count"] == 1
    alert_id = ouvertes["alertes"][0]["id"]
    assert client.post(f"/alertes/{alert_id}/acquitter").status_code == 200
    assert client.get("/alertes", params={"statut": "ouverte"}).json()["count"] == 0


def test_acquitter_alerte_inconnue(client, fake_db):
    assert client.post("/alertes/999/acquitter").status_code == 404


def test_monitoring(client, fake_db):
    envoyer(client, machine="USN-FR-01-M01")
    envoyer(client, machine="USN-FR-01-M02")
    assert client.get("/monitoring/recent").json()["count"] == 2
    assert client.get("/monitoring/machines").json()["count"] == 2


def test_webhook_sans_base(client, monkeypatch):
    monkeypatch.setattr(api, "DB_AVAILABLE", False)
    assert envoyer(client).status_code == 503
