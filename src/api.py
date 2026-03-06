"""
API REST pour la solution de maintenance prédictive MECHA.

Expose les modèles ML via des endpoints FastAPI :
- POST /predict/panne      : prédit si la machine est en panne (en_panne)
- POST /predict/panne24h   : prédit si une panne survient dans les 24 h
- POST /predict/rul        : prédit le temps restant avant défaillance (RUL)
- GET  /health             : état de l'API
- GET  /model/info         : métriques d'entraînement
"""

import json
import numpy as np
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="MECHA - API Maintenance Prédictive",
    description="API de prédiction de pannes sur le dataset unifié MECHA",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Chargement des modèles au démarrage
# ---------------------------------------------------------------------------

models = {}


def load_models():
    global models

    # Classificateur : en_panne
    for fname, key in [
        ("random_forest_classifier_mecha.joblib",     "clf_panne"),
        ("scaler_mecha.joblib",                        "scaler_panne"),
        ("feature_names_mecha.joblib",                 "feature_names"),
    ]:
        p = MODELS_DIR / fname
        if p.exists():
            models[key] = joblib.load(p)

    # Classificateur : panne_dans_24h
    for fname, key in [
        ("random_forest_classifier_mecha_24h.joblib", "clf_24h"),
        ("scaler_mecha_24h.joblib",                    "scaler_24h"),
    ]:
        p = MODELS_DIR / fname
        if p.exists():
            models[key] = joblib.load(p)

    # Régresseur RUL
    for fname, key in [
        ("random_forest_regressor_mecha.joblib",      "reg_rul"),
        ("scaler_rul_mecha.joblib",                    "scaler_rul"),
    ]:
        p = MODELS_DIR / fname
        if p.exists():
            models[key] = joblib.load(p)

    # Mappings catégoriels
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
    temperature_C:        float = Field(..., description="Température machine (°C)")
    vibration_mm_s:       float = Field(..., description="Vibration (mm/s)")
    courant_A:            float = Field(..., description="Courant électrique (A)")
    pression_bar:         float = Field(..., description="Pression hydraulique (bar)")
    vitesse_tr_min:       float = Field(..., description="Vitesse de rotation (tr/min)")
    age_machine_h:        int   = Field(..., description="Âge de la machine (heures)")
    h_depuis_maintenance: int   = Field(..., description="Heures depuis la dernière maintenance")
    type_machine:         str   = Field("CNC-Fraisage",
                                        description="Type : CNC-Fraisage, CNC-Tournage, Découpe-Laser, Centre-Usinage")
    usine_id:             str   = Field("USN-FR-01",
                                        description="Identifiant usine : USN-FR-01 … USN-ES-02")

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


# ---------------------------------------------------------------------------
# Construction du vecteur de features
# ---------------------------------------------------------------------------

def build_feature_vector(data: MechaPredictionRequest) -> np.ndarray:
    """
    Construit le vecteur de features dans le même ordre que FEATURE_COLS.
    Les moyennes 24 h et écarts-types sont approchés par les valeurs instantanées
    (pas d'historique disponible en prédiction ponctuelle).
    """
    maps = models.get("category_maps", {})
    type_enc  = maps.get("type_machine", {}).get(data.type_machine, 0)
    usine_enc = maps.get("usine_id",     {}).get(data.usine_id,     0)

    features = np.array([[
        # Capteurs instantanés
        data.temperature_C,
        data.vibration_mm_s,
        data.courant_A,
        data.pression_bar,
        data.vitesse_tr_min,
        # Contexte machine
        data.age_machine_h,
        data.h_depuis_maintenance,
        # Catégorielles
        type_enc,
        usine_enc,
    ]])
    return features


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
