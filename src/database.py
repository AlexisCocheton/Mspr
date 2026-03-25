"""
Couche d'accès à PostgreSQL pour MECHA — simulation production.

Utilise SQLAlchemy Core (pas d'ORM) pour rester cohérent avec
le style minimaliste du projet.

Variables d'environnement :
  DATABASE_URL  (défaut : postgresql://mecha:mecha@localhost:5432/mecha)
"""

import os
import logging
from sqlalchemy import (
    create_engine, Table, Column, MetaData,
    Integer, SmallInteger, String, Float, Boolean,
    DateTime, text,
)
from sqlalchemy.exc import OperationalError, SQLAlchemyError

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://mecha:mecha@localhost:5432/mecha",
)

# Engine module-level — create_engine() ne se connecte pas immédiatement.
# pool_pre_ping=True détecte et répare les connexions mortes au checkout.
engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

metadata = MetaData()

sensor_readings = Table(
    "sensor_readings", metadata,
    Column("id",                   Integer,    primary_key=True),
    Column("received_at",          DateTime(timezone=True)),
    Column("machine_id",           String(64),  nullable=False),
    Column("usine_id",             String(32),  nullable=False),
    Column("type_machine",         String(64),  nullable=False),
    Column("temperature_C",        Float,       nullable=False),
    Column("vibration_mm_s",       Float,       nullable=False),
    Column("courant_A",            Float,       nullable=False),
    Column("pression_bar",         Float,       nullable=False),
    Column("vitesse_tr_min",       Float,       nullable=False),
    Column("age_machine_h",        Integer,     nullable=False),
    Column("h_depuis_maintenance", Integer,     nullable=False),
)

predictions_table = Table(
    "predictions", metadata,
    Column("id",                   Integer,      primary_key=True),
    Column("reading_id",           Integer,      nullable=False),
    Column("predicted_at",         DateTime(timezone=True)),
    Column("machine_id",           String(64),   nullable=False),
    Column("usine_id",             String(32),   nullable=False),
    Column("type_machine",         String(64),   nullable=False),
    Column("panne24h_prediction",  SmallInteger, nullable=False),
    Column("panne24h_probability", Float,        nullable=False),
    Column("panne24h_risk_level",  String(16),   nullable=False),
    Column("rul_hours",            Float,        nullable=False),
    Column("rul_risk_level",       String(16),   nullable=False),
    Column("is_anomaly",           Boolean,      nullable=False),
    Column("anomaly_score",        Float,        nullable=False),
    Column("anomaly_risk_level",   String(16),   nullable=False),
)


def insert_reading(data: dict) -> int:
    """
    Insère une ligne dans sensor_readings.
    Clés attendues : machine_id, usine_id, type_machine,
                     temperature_C, vibration_mm_s, courant_A,
                     pression_bar, vitesse_tr_min,
                     age_machine_h, h_depuis_maintenance
    Retourne l'id généré.
    """
    with engine.connect() as conn:
        result = conn.execute(
            sensor_readings.insert().returning(sensor_readings.c.id),
            data,
        )
        conn.commit()
        return result.scalar_one()


def insert_prediction(
    reading_id: int,
    machine_id: str,
    usine_id: str,
    type_machine: str,
    panne24h: dict,
    rul: dict,
    anomaly: dict,
) -> int:
    """
    Insère une ligne dans predictions.
    panne24h : {prediction, probability, risk_level}
    rul      : {estimated_rul_hours, risk_level}
    anomaly  : {is_anomaly, anomaly_score, risk_level}
    Retourne l'id généré.
    """
    row = {
        "reading_id":           reading_id,
        "machine_id":           machine_id,
        "usine_id":             usine_id,
        "type_machine":         type_machine,
        "panne24h_prediction":  int(panne24h["prediction"]),
        "panne24h_probability": float(panne24h["probability"]),
        "panne24h_risk_level":  panne24h["risk_level"],
        "rul_hours":            float(rul["estimated_rul_hours"]),
        "rul_risk_level":       rul["risk_level"],
        "is_anomaly":           bool(anomaly["is_anomaly"]),
        "anomaly_score":        float(anomaly["anomaly_score"]),
        "anomaly_risk_level":   anomaly["risk_level"],
    }
    with engine.connect() as conn:
        result = conn.execute(
            predictions_table.insert().returning(predictions_table.c.id),
            row,
        )
        conn.commit()
        return result.scalar_one()


def get_recent_predictions(limit: int = 50) -> list[dict]:
    """Retourne les `limit` dernières prédictions, les plus récentes en premier."""
    with engine.connect() as conn:
        result = conn.execute(
            predictions_table.select()
            .order_by(predictions_table.c.predicted_at.desc())
            .limit(limit)
        )
        return [dict(row._mapping) for row in result]


def get_predictions_by_machine(machine_id: str, limit: int = 20) -> list[dict]:
    """Retourne les `limit` dernières prédictions pour une machine donnée."""
    with engine.connect() as conn:
        result = conn.execute(
            predictions_table.select()
            .where(predictions_table.c.machine_id == machine_id)
            .order_by(predictions_table.c.predicted_at.desc())
            .limit(limit)
        )
        return [dict(row._mapping) for row in result]


def get_last_prediction_per_machine() -> list[dict]:
    """
    Retourne la prédiction la plus récente pour chaque machine connue.
    Utilise DISTINCT ON (spécifique PostgreSQL) pour un accès en un seul scan.
    """
    sql = text("""
        SELECT DISTINCT ON (machine_id)
            machine_id, usine_id, type_machine,
            predicted_at,
            panne24h_prediction, panne24h_probability, panne24h_risk_level,
            rul_hours, rul_risk_level,
            is_anomaly, anomaly_score, anomaly_risk_level
        FROM predictions
        ORDER BY machine_id, predicted_at DESC
    """)
    with engine.connect() as conn:
        result = conn.execute(sql)
        return [dict(row._mapping) for row in result]


def is_db_available() -> bool:
    """Vérifie la connectivité à la base. Retourne False si indisponible."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except (OperationalError, SQLAlchemyError, Exception):
        return False
