"""
API REST pour la solution de maintenance prédictive MECHA.

Expose les modèles ML via des endpoints FastAPI :
- POST /predict/panne            : prédit si la machine est en panne (en_panne)
- POST /predict/panne24h         : prédit si une panne survient dans les 24 h
- POST /predict/rul              : prédit le temps restant avant défaillance (RUL)
- POST /predict/anomalie         : détecte un comportement capteur atypique (Isolation Forest)
- POST /predict/batch            : panne 24 h + RUL pour plusieurs machines (import CSV)
- POST /predict/panne/explain    : facteurs de la prédiction en_panne (SHAP)
- POST /predict/panne24h/explain : facteurs de la prédiction panne 24 h (SHAP)
- POST /webhook/sensor           : reçoit une lecture capteur, fait l'inférence,
                                   persiste en DB et déclenche les alertes
- GET  /monitoring/recent        : dernières prédictions depuis PostgreSQL
- GET  /monitoring/machines      : dernière prédiction par machine
- GET  /alertes                  : alertes (filtrables par statut)
- POST /alertes/{id}/acquitter   : acquittement d'une alerte par la maintenance
- GET  /config/regles            : règles métier actives (config/regles_metier.yaml)
- GET  /health                   : état de l'API
- GET  /model/info               : métriques d'entraînement

Les seuils, niveaux de risque et règles d'alerte proviennent de
config/regles_metier.yaml (module src.regles).

Les endpoints sont synchrones (def) : FastAPI les exécute dans un pool de
threads, ce qui évite de bloquer la boucle asynchrone pendant les appels
base de données et les inférences scikit-learn.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from src import regles

logger = logging.getLogger(__name__)

# Import conditionnel de la couche DB — les endpoints de prédiction fonctionnent sans Postgres
try:
    from src import database as db
    DB_AVAILABLE = True
except Exception as _db_exc:
    logger.warning("Module database indisponible (%s) — webhook/monitoring désactivés.", _db_exc)
    db = None
    DB_AVAILABLE = False

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

models = {}


def load_models():
    for fname, key in [
        ("random_forest_classifier_mecha.joblib",     "clf_panne"),
        ("scaler_mecha.joblib",                        "scaler_panne"),
        ("feature_names_mecha.joblib",                 "feature_names"),
        ("random_forest_classifier_mecha_24h.joblib", "clf_24h"),
        ("scaler_mecha_24h.joblib",                    "scaler_24h"),
        ("random_forest_regressor_mecha.joblib",      "reg_rul"),
        ("scaler_rul_mecha.joblib",                    "scaler_rul"),
        ("isolation_forest_mecha.joblib",              "iso_forest"),
        ("scaler_anomaly_mecha.joblib",                "scaler_anomaly"),
    ]:
        p = MODELS_DIR / fname
        if p.exists():
            models[key] = joblib.load(p)

    maps_path = BASE_DIR / "data" / "processed" / "category_maps.json"
    if maps_path.exists():
        with open(maps_path, encoding="utf-8") as f:
            models["category_maps"] = json.load(f)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    regles.charger_regles()          # échoue au démarrage si le fichier de règles est invalide
    load_models()
    if DB_AVAILABLE:
        try:
            db.init_schema()
        except Exception as exc:
            logger.warning("Initialisation du schéma impossible (%s) — base indisponible ?", exc)
    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MECHA - API Maintenance Prédictive",
    description="API de prédiction de pannes sur le dataset unifié MECHA",
    version="4.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schémas
# ---------------------------------------------------------------------------

_BORNES = regles.charger_regles()["capteurs"]


def _champ(nom: str, description: str):
    """Champ Pydantic borné par les limites physiques du fichier de règles."""
    b = _BORNES[nom]
    return Field(..., ge=b["min"], le=b["max"], description=f"{description} ({b['unite']})")


class MechaPredictionRequest(BaseModel):
    """Données capteurs d'une machine MECHA."""
    temperature_C:        float = _champ("temperature_C",        "Température machine")
    vibration_mm_s:       float = _champ("vibration_mm_s",       "Vibration")
    courant_A:            float = _champ("courant_A",            "Courant électrique")
    pression_bar:         float = _champ("pression_bar",         "Pression hydraulique")
    vitesse_tr_min:       float = _champ("vitesse_tr_min",       "Vitesse de rotation")
    age_machine_h:        int   = _champ("age_machine_h",        "Âge de la machine")
    h_depuis_maintenance: int   = _champ("h_depuis_maintenance", "Heures depuis la dernière maintenance")
    type_machine: Literal["CNC-Fraisage", "CNC-Tournage", "Découpe-Laser", "Centre-Usinage"] = Field(
        "CNC-Fraisage", description="Type de machine"
    )
    usine_id: Literal["USN-FR-01", "USN-FR-02", "USN-FR-03", "USN-ES-01", "USN-ES-02"] = Field(
        "USN-FR-01", description="Identifiant usine"
    )

    model_config = {"json_schema_extra": {"examples": [{
        "temperature_C": 72.0,
        "vibration_mm_s": 2.8,
        "courant_A": 20.5,
        "pression_bar": 5.9,
        "vitesse_tr_min": 1380,
        "age_machine_h": 15000,
        "h_depuis_maintenance": 600,
        "type_machine": "CNC-Fraisage",
        "usine_id": "USN-FR-01",
    }]}}


class PredictionResponse(BaseModel):
    prediction:     int   = Field(..., description="0 = normal, 1 = panne/imminente")
    probability:    float = Field(..., description="Probabilité de panne")
    risk_level:     str   = Field(..., description="faible / moyen / élevé / critique")
    recommendation: str


class RULResponse(BaseModel):
    estimated_rul_hours: float
    risk_level:          str
    recommendation:      str


class AnomalyResponse(BaseModel):
    is_anomaly:     bool  = Field(..., description="True si comportement atypique détecté")
    anomaly_score:  float = Field(..., description="Score d'anomalie normalisé [0–1], 1 = très anormal")
    risk_level:     str   = Field(..., description="normal / suspect / anomalie")
    recommendation: str


class FeatureContribution(BaseModel):
    feature:      str
    value:        float
    contribution: float
    direction:    str = Field(..., description="hausse_risque / baisse_risque")


class ExplainResponse(BaseModel):
    prediction:     int
    probability:    float
    risk_level:     str
    recommendation: str
    top_features:   list[FeatureContribution]
    methode:        str = Field(..., description="shap / importance_globale")


class BatchRequest(BaseModel):
    lignes: list[MechaPredictionRequest] = Field(..., min_length=1, max_length=1000)


class BatchLigne(BaseModel):
    panne24h: PredictionResponse
    rul:      RULResponse | None = None


class BatchResponse(BaseModel):
    resultats: list[BatchLigne]


# ---------------------------------------------------------------------------
# Inférence
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "temperature_C", "vibration_mm_s", "courant_A",
    "pression_bar", "vitesse_tr_min",
    "age_machine_h", "h_depuis_maintenance",
    "type_machine_encoded", "usine_encoded",
]


def build_feature_matrix(lignes: list[MechaPredictionRequest]) -> np.ndarray:
    maps = models.get("category_maps", {})
    return np.array([[
        d.temperature_C, d.vibration_mm_s, d.courant_A,
        d.pression_bar,  d.vitesse_tr_min,
        d.age_machine_h, d.h_depuis_maintenance,
        maps.get("type_machine", {}).get(d.type_machine, 0),
        maps.get("usine_id",     {}).get(d.usine_id,     0),
    ] for d in lignes], dtype=float)


def build_feature_vector(data: MechaPredictionRequest) -> np.ndarray:
    return build_feature_matrix([data])


def _exiger(*cles: str) -> None:
    manquants = [k for k in cles if k not in models]
    if manquants:
        raise HTTPException(503, f"Modèles non chargés : {manquants}. Lancez model_training.py.")


def _classifier(features: np.ndarray, clf_key: str, scaler_key: str) -> list[PredictionResponse]:
    X = models[scaler_key].transform(features)
    preds  = models[clf_key].predict(X)
    probas = models[clf_key].predict_proba(X)[:, 1]
    out = []
    for pred, proba in zip(preds, probas):
        niveau, action = regles.niveau_panne(float(proba))
        out.append(PredictionResponse(
            prediction=int(pred), probability=round(float(proba), 4),
            risk_level=niveau, recommendation=action,
        ))
    return out


def _rul(features: np.ndarray) -> list[RULResponse]:
    ruls = models["reg_rul"].predict(models["scaler_rul"].transform(features))
    out = []
    for rul in ruls:
        rul = float(max(0.0, rul))
        niveau, action = regles.niveau_rul(rul)
        out.append(RULResponse(estimated_rul_hours=round(rul, 2), risk_level=niveau, recommendation=action))
    return out


def _anomalie(features: np.ndarray) -> AnomalyResponse:
    X = models["scaler_anomaly"].transform(features)
    est_anomalie = bool(models["iso_forest"].predict(X)[0] == -1)    # -1 = anomalie
    score = regles.score_anomalie_normalise(float(models["iso_forest"].score_samples(X)[0]))
    niveau, action = regles.niveau_anomalie(est_anomalie, score)
    return AnomalyResponse(
        is_anomaly=est_anomalie, anomaly_score=round(score, 4),
        risk_level=niveau, recommendation=action,
    )


def _contributions(clf_key: str, features_scaled: np.ndarray, prediction: int) -> tuple[np.ndarray, str]:
    """Contributions de chaque feature à la probabilité de panne (classe 1)."""
    clf = models[clf_key]
    try:
        import shap
        cache_key = f"explainer_{clf_key}"
        if cache_key not in models:
            models[cache_key] = shap.TreeExplainer(clf)
        sv = models[cache_key].shap_values(features_scaled)
        if isinstance(sv, list):              # shap < 0.45 : une matrice par classe
            contribs = np.asarray(sv[1])[0]
        else:
            sv = np.asarray(sv)               # shap >= 0.45 : (n, features[, classes])
            contribs = sv[0, :, 1] if sv.ndim == 3 else sv[0]
        return np.asarray(contribs, dtype=float), "shap"
    except ImportError:
        # Repli : importances globales orientées selon la prédiction
        sign = 1 if prediction == 1 else -1
        return clf.feature_importances_ * sign, "importance_globale"


def _expliquer(data: MechaPredictionRequest, clf_key: str, scaler_key: str) -> ExplainResponse:
    features = build_feature_vector(data)
    resultat = _classifier(features, clf_key, scaler_key)[0]
    contribs, methode = _contributions(
        clf_key, models[scaler_key].transform(features), resultat.prediction
    )
    top = np.argsort(np.abs(contribs))[::-1][:3]
    return ExplainResponse(
        **resultat.model_dump(),
        methode=methode,
        top_features=[
            FeatureContribution(
                feature=FEATURE_NAMES[i],
                value=round(float(features[0][i]), 3),
                contribution=round(float(contribs[i]), 4),
                direction="hausse_risque" if contribs[i] > 0 else "baisse_risque",
            )
            for i in top
        ],
    )


# ---------------------------------------------------------------------------
# Endpoints — état et prédiction
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": [k for k in models if not k.startswith("explainer_")],
        "database": bool(DB_AVAILABLE and db.is_db_available()),
    }


@app.get("/model/info")
def model_info():
    info = {"available_models": [k for k in models if not k.startswith("explainer_")]}
    results_path = MODELS_DIR / "training_results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            info["training_results"] = json.load(f)
    return info


@app.get("/config/regles")
def config_regles():
    """Règles métier actives : seuils, niveaux, déclenchement des alertes, fréquences."""
    return regles.charger_regles()


@app.post("/predict/panne", response_model=PredictionResponse)
def predict_panne(data: MechaPredictionRequest):
    """Prédit si la machine est actuellement en panne."""
    _exiger("clf_panne", "scaler_panne")
    return _classifier(build_feature_vector(data), "clf_panne", "scaler_panne")[0]


@app.post("/predict/panne24h", response_model=PredictionResponse)
def predict_panne_24h(data: MechaPredictionRequest):
    """Prédit si une panne surviendra dans les 24 heures."""
    _exiger("clf_24h", "scaler_24h")
    return _classifier(build_feature_vector(data), "clf_24h", "scaler_24h")[0]


@app.post("/predict/rul", response_model=RULResponse)
def predict_rul(data: MechaPredictionRequest):
    """Prédit le temps restant avant la prochaine défaillance (RUL)."""
    _exiger("reg_rul", "scaler_rul")
    return _rul(build_feature_vector(data))[0]


@app.post("/predict/anomalie", response_model=AnomalyResponse)
def predict_anomalie(data: MechaPredictionRequest):
    """
    Détecte un comportement capteur atypique via Isolation Forest.
    Approche non supervisée — ne nécessite pas de labels de panne.
    Complémentaire aux classifieurs supervisés.
    """
    _exiger("iso_forest", "scaler_anomaly")
    return _anomalie(build_feature_vector(data))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    """Panne 24 h (+ RUL si disponible) pour une liste de relevés (1 000 maximum)."""
    _exiger("clf_24h", "scaler_24h")
    features = build_feature_matrix(req.lignes)
    pannes = _classifier(features, "clf_24h", "scaler_24h")
    ruls = _rul(features) if "reg_rul" in models else [None] * len(pannes)
    return BatchResponse(resultats=[BatchLigne(panne24h=p, rul=r) for p, r in zip(pannes, ruls)])


@app.post("/predict/panne/explain", response_model=ExplainResponse)
def explain_panne(data: MechaPredictionRequest):
    """Facteurs ayant conduit à la prédiction en_panne (SHAP TreeExplainer)."""
    _exiger("clf_panne", "scaler_panne")
    return _expliquer(data, "clf_panne", "scaler_panne")


@app.post("/predict/panne24h/explain", response_model=ExplainResponse)
def explain_panne_24h(data: MechaPredictionRequest):
    """Facteurs ayant conduit à la prédiction de panne dans les 24 h (SHAP TreeExplainer)."""
    _exiger("clf_24h", "scaler_24h")
    return _expliquer(data, "clf_24h", "scaler_24h")


# ---------------------------------------------------------------------------
# Webhook + alertes (simulation production avec PostgreSQL)
# ---------------------------------------------------------------------------

class WebhookSensorRequest(MechaPredictionRequest):
    """Étend la requête de prédiction avec l'identité physique de la machine."""
    machine_id:  str = Field(..., description="Identifiant unique de la machine (ex: USN-FR-01-M01)")
    measured_at: datetime | None = Field(
        None, description="Horodatage de la mesure à la source (capteur / SCADA)"
    )


class AlertResponse(BaseModel):
    id:          int
    machine_id:  str
    source:      str
    niveau:      str
    message:     str
    notifiee:    bool


class WebhookResponse(BaseModel):
    reading_id:    int
    prediction_id: int
    machine_id:    str
    panne24h:      PredictionResponse
    rul:           RULResponse
    anomaly:       AnomalyResponse
    alertes:       list[AlertResponse] = []


def _exiger_db() -> None:
    if not DB_AVAILABLE:
        raise HTTPException(503, "Base de données non disponible (DB_AVAILABLE=False).")


def notifier(alerte: dict) -> bool:
    """
    Journalise l'alerte et, si une URL est configurée, la pousse en POST JSON
    (outil de ticketing, SCADA, messagerie…). Retourne True si la notification
    externe a abouti.
    """
    cfg = regles.charger_regles()["alertes"]["notification"]
    logger.warning("ALERTE %s — %s (%s) : %s",
                   alerte["niveau"].upper(), alerte["machine_id"], alerte["source"], alerte["message"])
    url = os.getenv("MECHA_ALERT_WEBHOOK_URL") or cfg.get("webhook_url")
    if not url:
        return False
    try:
        r = httpx.post(url, json=jsonable_encoder(alerte), timeout=float(cfg.get("timeout_s", 3)))
        r.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("Notification de l'alerte %s échouée : %s", alerte["id"], exc)
        return False
    db.mark_alert_notified(alerte["id"])
    return True


def evaluer_alertes(data: WebhookSensorRequest, prediction_id: int,
                    panne24h: PredictionResponse, rul: RULResponse,
                    anomaly: AnomalyResponse) -> list[AlertResponse]:
    """
    Applique les règles d'alerte de config/regles_metier.yaml après chaque
    prédiction : N lectures consécutives dans un niveau déclencheur, et
    aucune alerte déjà ouverte pour la même machine et la même source.
    """
    creees = []
    for source, resultat in (("panne24h", panne24h), ("anomalie", anomaly)):
        n = regles.lectures_requises(source)
        niveaux = db.get_recent_levels(data.machine_id, source, n)
        if not regles.doit_alerter(source, niveaux):
            continue
        if db.has_open_alert(data.machine_id, source):
            continue
        alerte = db.insert_alert({
            "prediction_id": prediction_id,
            "machine_id":    data.machine_id,
            "usine_id":      data.usine_id,
            "type_machine":  data.type_machine,
            "source":        source,
            "niveau":        resultat.risk_level,
            "probabilite":   panne24h.probability if source == "panne24h" else None,
            "rul_hours":     rul.estimated_rul_hours,
            "message":       f"Niveau « {resultat.risk_level} » sur {n} lectures consécutives. "
                             f"{resultat.recommendation}",
        })
        alerte["notifiee"] = notifier(alerte)
        creees.append(AlertResponse(**{k: alerte[k] for k in AlertResponse.model_fields}))
    return creees


@app.post("/webhook/sensor", response_model=WebhookResponse)
def webhook_sensor(data: WebhookSensorRequest):
    """
    Reçoit une lecture capteur, exécute les 3 inférences ML (panne24h, RUL, anomalie),
    persiste la lecture + la prédiction dans PostgreSQL et déclenche les alertes.
    """
    _exiger_db()
    _exiger("clf_24h", "scaler_24h", "reg_rul", "scaler_rul", "iso_forest", "scaler_anomaly")

    features = build_feature_vector(data)   # shape (1, 9)

    reading_payload = {
        "machine_id":           data.machine_id,
        "usine_id":             data.usine_id,
        "type_machine":         data.type_machine,
        "measured_at":          data.measured_at,
        "temperature_C":        data.temperature_C,
        "vibration_mm_s":       data.vibration_mm_s,
        "courant_A":            data.courant_A,
        "pression_bar":         data.pression_bar,
        "vitesse_tr_min":       data.vitesse_tr_min,
        "age_machine_h":        data.age_machine_h,
        "h_depuis_maintenance": data.h_depuis_maintenance,
    }
    try:
        reading_id = db.insert_reading(reading_payload)
    except Exception as exc:
        logger.error("Erreur insert_reading : %s", exc)
        raise HTTPException(500, "Erreur d'écriture en base de données.")

    panne24h_result = _classifier(features, "clf_24h", "scaler_24h")[0]
    rul_result      = _rul(features)[0]
    anomaly_result  = _anomalie(features)

    try:
        prediction_id = db.insert_prediction(
            reading_id=reading_id,
            machine_id=data.machine_id,
            usine_id=data.usine_id,
            type_machine=data.type_machine,
            panne24h=panne24h_result.model_dump(),
            rul=rul_result.model_dump(),
            anomaly=anomaly_result.model_dump(),
        )
    except Exception as exc:
        logger.error("Erreur insert_prediction : %s", exc)
        raise HTTPException(500, "Erreur d'écriture prédiction en base de données.")

    # Une erreur sur les alertes ne doit pas faire perdre la prédiction enregistrée
    try:
        alertes = evaluer_alertes(data, prediction_id, panne24h_result, rul_result, anomaly_result)
    except Exception as exc:
        logger.error("Évaluation des alertes impossible : %s", exc)
        alertes = []

    return WebhookResponse(
        reading_id=reading_id,
        prediction_id=prediction_id,
        machine_id=data.machine_id,
        panne24h=panne24h_result,
        rul=rul_result,
        anomaly=anomaly_result,
        alertes=alertes,
    )


def _serialiser_dates(rows: list[dict]) -> list[dict]:
    for row in rows:
        for k, v in row.items():
            if isinstance(v, datetime):
                row[k] = v.isoformat()
    return rows


@app.get("/monitoring/recent")
def monitoring_recent(limit: int = 50):
    """Retourne les N dernières prédictions depuis PostgreSQL."""
    _exiger_db()
    try:
        rows = _serialiser_dates(db.get_recent_predictions(limit=min(limit, 200)))
        return {"predictions": rows, "count": len(rows)}
    except Exception as exc:
        logger.error("monitoring/recent : %s", exc)
        raise HTTPException(500, "Erreur de lecture en base de données.")


@app.get("/monitoring/machines")
def monitoring_machines():
    """Retourne la dernière prédiction par machine (DISTINCT ON)."""
    _exiger_db()
    try:
        rows = _serialiser_dates(db.get_last_prediction_per_machine())
        return {"machines": rows, "count": len(rows)}
    except Exception as exc:
        logger.error("monitoring/machines : %s", exc)
        raise HTTPException(500, "Erreur de lecture en base de données.")


@app.get("/alertes")
def lister_alertes(statut: Literal["ouverte", "acquittee"] | None = None, limit: int = 50):
    """Alertes les plus récentes en premier (statut : ouverte / acquittee)."""
    _exiger_db()
    try:
        rows = _serialiser_dates(db.get_alerts(statut=statut, limit=min(limit, 500)))
        return {"alertes": rows, "count": len(rows)}
    except Exception as exc:
        logger.error("alertes : %s", exc)
        raise HTTPException(500, "Erreur de lecture en base de données.")


@app.post("/alertes/{alert_id}/acquitter")
def acquitter_alerte(alert_id: int):
    """Acquitte une alerte : la maintenance l'a prise en charge."""
    _exiger_db()
    try:
        row = db.acknowledge_alert(alert_id)
    except Exception as exc:
        logger.error("acquittement : %s", exc)
        raise HTTPException(500, "Erreur d'écriture en base de données.")
    if row is None:
        raise HTTPException(404, f"Alerte {alert_id} introuvable.")
    return _serialiser_dates([row])[0]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
