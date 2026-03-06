"""
API REST pour la solution de maintenance prédictive MECHA.

Expose les modèles ML via des endpoints FastAPI :
- POST /predict/panne        : prédit si la machine est en panne (en_panne)
- POST /predict/panne24h     : prédit si une panne survient dans les 24 h
- POST /predict/rul          : prédit le temps restant avant défaillance (RUL)
- POST /predict/anomalie     : détecte un comportement capteur atypique (Isolation Forest)
- POST /predict/panne/explain: explique les facteurs de la prédiction panne (SHAP)
- GET  /health               : état de l'API
- GET  /model/info           : métriques d'entraînement
"""

import json
import numpy as np
import joblib
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="MECHA - API Maintenance Prédictive",
    description="API de prédiction de pannes sur le dataset unifié MECHA",
    version="3.0.0",
)

# ---------------------------------------------------------------------------
# Chargement des modèles au démarrage
# ---------------------------------------------------------------------------

models = {}


def load_models():
    global models

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


@app.on_event("startup")
async def startup_event():
    load_models()


# ---------------------------------------------------------------------------
# Schémas
# ---------------------------------------------------------------------------

class MechaPredictionRequest(BaseModel):
    """Données capteurs d'une machine MECHA."""
    temperature_C:        float = Field(..., ge=-20,    le=300,    description="Température machine (°C)")
    vibration_mm_s:       float = Field(..., ge=0,      le=20,     description="Vibration (mm/s)")
    courant_A:            float = Field(..., ge=0,      le=100,    description="Courant électrique (A)")
    pression_bar:         float = Field(..., ge=0,      le=20,     description="Pression hydraulique (bar)")
    vitesse_tr_min:       float = Field(..., ge=0,      le=3000,   description="Vitesse de rotation (tr/min)")
    age_machine_h:        int   = Field(..., ge=0,      le=200000, description="Âge de la machine (heures)")
    h_depuis_maintenance: int   = Field(..., ge=0,      le=10000,  description="Heures depuis la dernière maintenance")
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


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "temperature_C", "vibration_mm_s", "courant_A",
    "pression_bar", "vitesse_tr_min",
    "age_machine_h", "h_depuis_maintenance",
    "type_machine_encoded", "usine_encoded",
]


def build_feature_vector(data: MechaPredictionRequest) -> np.ndarray:
    maps      = models.get("category_maps", {})
    type_enc  = maps.get("type_machine", {}).get(data.type_machine, 0)
    usine_enc = maps.get("usine_id",     {}).get(data.usine_id,     0)
    return np.array([[
        data.temperature_C, data.vibration_mm_s, data.courant_A,
        data.pression_bar,  data.vitesse_tr_min,
        data.age_machine_h, data.h_depuis_maintenance,
        type_enc, usine_enc,
    ]])


def get_risk_assessment(probability: float) -> tuple[str, str]:
    if probability >= 0.8:
        return "critique", "ARRÊT IMMÉDIAT recommandé. Intervention urgente requise."
    elif probability >= 0.5:
        return "élevé",    "Planifier une maintenance préventive dans les prochaines heures."
    elif probability >= 0.3:
        return "moyen",    "Surveillance renforcée. Vérifier les paramètres machine."
    else:
        return "faible",   "Machine en fonctionnement normal. Aucune action requise."


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.get("/model/info")
async def model_info():
    info = {"available_models": list(models.keys())}
    results_path = MODELS_DIR / "training_results.json"
    if results_path.exists():
        with open(results_path) as f:
            info["training_results"] = json.load(f)
    return info


@app.post("/predict/panne", response_model=PredictionResponse)
async def predict_panne(data: MechaPredictionRequest):
    """Prédit si la machine est actuellement en panne."""
    if "clf_panne" not in models:
        raise HTTPException(503, "Modèle en_panne non chargé. Lancez model_training.py.")

    features        = build_feature_vector(data)
    features_scaled = models["scaler_panne"].transform(features)
    prediction      = int(models["clf_panne"].predict(features_scaled)[0])
    probability     = float(models["clf_panne"].predict_proba(features_scaled)[0][1])
    risk, reco      = get_risk_assessment(probability)

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=risk,
        recommendation=reco,
    )


@app.post("/predict/panne24h", response_model=PredictionResponse)
async def predict_panne_24h(data: MechaPredictionRequest):
    """Prédit si une panne surviendra dans les 24 heures."""
    if "clf_24h" not in models:
        raise HTTPException(503, "Modèle panne_24h non chargé. Lancez model_training.py.")

    features        = build_feature_vector(data)
    features_scaled = models["scaler_24h"].transform(features)
    prediction      = int(models["clf_24h"].predict(features_scaled)[0])
    probability     = float(models["clf_24h"].predict_proba(features_scaled)[0][1])
    risk, reco      = get_risk_assessment(probability)

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=risk,
        recommendation=reco,
    )


@app.post("/predict/rul", response_model=RULResponse)
async def predict_rul(data: MechaPredictionRequest):
    """Prédit le temps restant avant la prochaine défaillance (RUL)."""
    if "reg_rul" not in models:
        raise HTTPException(503, "Modèle RUL non chargé. Lancez model_training.py.")

    features        = build_feature_vector(data)
    features_scaled = models["scaler_rul"].transform(features)
    rul             = float(max(0, models["reg_rul"].predict(features_scaled)[0]))

    if rul < 6:
        risk = "critique"
        reco = "ARRÊT IMMÉDIAT. Maintenance urgente à planifier."
    elif rul < 12:
        risk = "élevé"
        reco = "Maintenance à prévoir dans les prochaines heures."
    elif rul < 24:
        risk = "moyen"
        reco = "Surveiller de près. Maintenance sous 24 h."
    else:
        risk = "faible"
        reco = "Machine en état normal. Prochaine maintenance planifiée."

    return RULResponse(
        estimated_rul_hours=round(rul, 2),
        risk_level=risk,
        recommendation=reco,
    )


@app.post("/predict/anomalie", response_model=AnomalyResponse)
async def predict_anomalie(data: MechaPredictionRequest):
    """
    Détecte un comportement capteur atypique via Isolation Forest.
    Approche non supervisée — ne nécessite pas de labels de panne.
    Complémentaire aux classifieurs supervisés.
    """
    if "iso_forest" not in models:
        raise HTTPException(503, "Modèle Isolation Forest non chargé. Lancez model_training.py.")

    features        = build_feature_vector(data)
    features_scaled = models["scaler_anomaly"].transform(features)

    pred  = models["iso_forest"].predict(features_scaled)[0]       # -1 = anomalie, +1 = normal
    score = float(models["iso_forest"].score_samples(features_scaled)[0])

    is_anomaly = pred == -1
    # score_samples est négatif ; on normalise vers [0, 1] (1 = très anormal)
    normalized = float(max(0.0, min(1.0, (-score - 0.3) / 0.4)))

    if is_anomaly and normalized > 0.7:
        risk = "anomalie"
        reco = "Comportement capteur très inhabituel. Vérifier la machine immédiatement."
    elif is_anomaly:
        risk = "suspect"
        reco = "Comportement légèrement atypique. Surveillance renforcée recommandée."
    else:
        risk = "normal"
        reco = "Comportement capteur dans les normes habituelles."

    return AnomalyResponse(
        is_anomaly=is_anomaly,
        anomaly_score=round(normalized, 4),
        risk_level=risk,
        recommendation=reco,
    )


@app.post("/predict/panne/explain", response_model=ExplainResponse)
async def explain_panne(data: MechaPredictionRequest):
    """
    Explique les facteurs ayant conduit à la prédiction de panne.
    Utilise SHAP TreeExplainer si disponible, sinon les importances globales RF.
    """
    if "clf_panne" not in models:
        raise HTTPException(503, "Modèle en_panne non chargé. Lancez model_training.py.")

    features        = build_feature_vector(data)
    features_scaled = models["scaler_panne"].transform(features)
    prediction      = int(models["clf_panne"].predict(features_scaled)[0])
    probability     = float(models["clf_panne"].predict_proba(features_scaled)[0][1])
    risk, reco      = get_risk_assessment(probability)

    try:
        import shap
        explainer   = shap.TreeExplainer(models["clf_panne"])
        shap_values = explainer.shap_values(features_scaled)
        contribs    = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    except ImportError:
        # Fallback : importances globales orientées selon la prédiction
        sign    = 1 if prediction == 1 else -1
        contribs = models["clf_panne"].feature_importances_ * sign

    feat_values  = features[0]
    top_indices  = np.argsort(np.abs(contribs))[::-1][:3]

    top_features = [
        FeatureContribution(
            feature=FEATURE_NAMES[i],
            value=round(float(feat_values[i]), 3),
            contribution=round(float(contribs[i]), 4),
            direction="hausse_risque" if contribs[i] > 0 else "baisse_risque",
        )
        for i in top_indices
    ]

    return ExplainResponse(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=risk,
        recommendation=reco,
        top_features=top_features,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
