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
    create_engine, Table, Column, MetaData, ForeignKey, Index,
    Integer, SmallInteger, String, Float, Boolean, Text,
    DateTime, func, text,
)
from sqlalchemy.exc import OperationalError, SQLAlchemyError

logger = logging.getLogger(__name__)


def _normaliser_url(url: str) -> str:
    """Force le pilote psycopg2 (SQLAlchemy >= 2.1 choisit psycopg v3 par défaut)."""
    if url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url


DATABASE_URL = _normaliser_url(os.getenv(
    "DATABASE_URL",
    "postgresql://mecha:mecha@localhost:5432/mecha",
))

# Engine module-level — create_engine() ne se connecte pas immédiatement.
# pool_pre_ping=True détecte et répare les connexions mortes au checkout.
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args={"connect_timeout": 3})

metadata = MetaData()

sensor_readings = Table(
    "sensor_readings", metadata,
    Column("id",                   Integer,    primary_key=True),
    Column("received_at",          DateTime(timezone=True), server_default=func.now()),
    # Horodatage de la mesure à la source (capteur / SCADA), distinct de
    # l'heure de réception : indispensable pour ordonner et synchroniser les flux.
    Column("measured_at",          DateTime(timezone=True), nullable=True),
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
    Column("reading_id",           Integer,      ForeignKey("sensor_readings.id", ondelete="CASCADE"),
           nullable=False),
    Column("predicted_at",         DateTime(timezone=True), server_default=func.now()),
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
    Index("idx_predictions_machine_time", "machine_id", "predicted_at"),
    Index("idx_predictions_time", "predicted_at"),
)

alerts_table = Table(
    "alertes", metadata,
    Column("id",            Integer,     primary_key=True),
    Column("created_at",    DateTime(timezone=True), server_default=func.now()),
    Column("prediction_id", Integer,     ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True),
    Column("machine_id",    String(64),  nullable=False),
    Column("usine_id",      String(32),  nullable=False),
    Column("type_machine",  String(64),  nullable=False),
    Column("source",        String(16),  nullable=False),   # panne24h | anomalie
    Column("niveau",        String(16),  nullable=False),
    Column("probabilite",   Float,       nullable=True),
    Column("rul_hours",     Float,       nullable=True),
    Column("message",       Text,        nullable=False),
    Column("statut",        String(16),  nullable=False, server_default="ouverte"),  # ouverte | acquittee
    Column("acquittee_at",  DateTime(timezone=True), nullable=True),
    Column("notifiee",      Boolean,     nullable=False, server_default="false"),
    Index("idx_alertes_statut_time", "statut", "created_at"),
    Index("idx_alertes_machine_source", "machine_id", "source", "statut"),
)

# Colonne de prédiction suivie pour chaque source d'alerte
_COLONNE_NIVEAU = {
    "panne24h": predictions_table.c.panne24h_risk_level,
    "anomalie": predictions_table.c.anomaly_risk_level,
}


def init_schema() -> None:
    """
    Crée les tables manquantes et ajoute les colonnes apparues depuis la
    première version du schéma (base déjà initialisée par db/init.sql).
    """
    metadata.create_all(engine, checkfirst=True)
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE sensor_readings ADD COLUMN IF NOT EXISTS measured_at TIMESTAMPTZ"
        ))


def insert_reading(data: dict) -> int:
    """
    Insère une ligne dans sensor_readings.
    Clés attendues : machine_id, usine_id, type_machine,
                     temperature_C, vibration_mm_s, courant_A,
                     pression_bar, vitesse_tr_min,
                     age_machine_h, h_depuis_maintenance
    Clé optionnelle : measured_at
    Retourne l'id généré.
    """
    with engine.begin() as conn:
        result = conn.execute(
            sensor_readings.insert().returning(sensor_readings.c.id),
            data,
        )
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
    with engine.begin() as conn:
        result = conn.execute(
            predictions_table.insert().returning(predictions_table.c.id),
            row,
        )
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


# ---------------------------------------------------------------------------
# Alertes
# ---------------------------------------------------------------------------

def get_recent_levels(machine_id: str, source: str, n: int) -> list[str]:
    """Niveaux de risque des `n` dernières prédictions d'une machine (plus récent en premier)."""
    col = _COLONNE_NIVEAU[source]
    with engine.connect() as conn:
        result = conn.execute(
            predictions_table.select().with_only_columns(col)
            .where(predictions_table.c.machine_id == machine_id)
            .order_by(predictions_table.c.id.desc())
            .limit(n)
        )
        return [row[0] for row in result]


def has_open_alert(machine_id: str, source: str) -> bool:
    """True si une alerte non acquittée existe déjà pour cette machine et cette source."""
    with engine.connect() as conn:
        result = conn.execute(
            alerts_table.select().with_only_columns(alerts_table.c.id)
            .where(alerts_table.c.machine_id == machine_id)
            .where(alerts_table.c.source == source)
            .where(alerts_table.c.statut == "ouverte")
            .limit(1)
        )
        return result.first() is not None


def insert_alert(data: dict) -> dict:
    """Insère une alerte et retourne la ligne créée."""
    with engine.begin() as conn:
        result = conn.execute(alerts_table.insert().returning(*alerts_table.c), data)
        return dict(result.one()._mapping)


def mark_alert_notified(alert_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(
            alerts_table.update().where(alerts_table.c.id == alert_id).values(notifiee=True)
        )


def get_alerts(statut: str | None = None, limit: int = 50) -> list[dict]:
    """Alertes les plus récentes en premier, filtrables par statut."""
    query = alerts_table.select().order_by(alerts_table.c.created_at.desc()).limit(limit)
    if statut:
        query = query.where(alerts_table.c.statut == statut)
    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(query)]


def acknowledge_alert(alert_id: int) -> dict | None:
    """Acquitte une alerte ouverte. Retourne la ligne mise à jour, ou None si introuvable."""
    with engine.begin() as conn:
        result = conn.execute(
            alerts_table.update()
            .where(alerts_table.c.id == alert_id)
            .values(statut="acquittee", acquittee_at=func.now())
            .returning(*alerts_table.c)
        )
        row = result.first()
        return dict(row._mapping) if row else None


def is_db_available() -> bool:
    """Vérifie la connectivité à la base. Retourne False si indisponible."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except (OperationalError, SQLAlchemyError, Exception):
        return False
