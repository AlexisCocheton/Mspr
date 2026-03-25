-- MECHA — Schéma PostgreSQL pour la simulation production
-- Exécuté automatiquement au premier démarrage du container postgres

CREATE TABLE IF NOT EXISTS sensor_readings (
    id                    SERIAL PRIMARY KEY,
    received_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    machine_id            VARCHAR(64)  NOT NULL,
    usine_id              VARCHAR(32)  NOT NULL,
    type_machine          VARCHAR(64)  NOT NULL,
    "temperature_C"       REAL         NOT NULL,
    vibration_mm_s        REAL         NOT NULL,
    "courant_A"           REAL         NOT NULL,
    pression_bar          REAL         NOT NULL,
    vitesse_tr_min        REAL         NOT NULL,
    age_machine_h         INTEGER      NOT NULL,
    h_depuis_maintenance  INTEGER      NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id                    SERIAL PRIMARY KEY,
    reading_id            INTEGER      NOT NULL REFERENCES sensor_readings(id) ON DELETE CASCADE,
    predicted_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    machine_id            VARCHAR(64)  NOT NULL,
    usine_id              VARCHAR(32)  NOT NULL,
    type_machine          VARCHAR(64)  NOT NULL,
    panne24h_prediction   SMALLINT     NOT NULL,
    panne24h_probability  REAL         NOT NULL,
    panne24h_risk_level   VARCHAR(16)  NOT NULL,
    rul_hours             REAL         NOT NULL,
    rul_risk_level        VARCHAR(16)  NOT NULL,
    is_anomaly            BOOLEAN      NOT NULL,
    anomaly_score         REAL         NOT NULL,
    anomaly_risk_level    VARCHAR(16)  NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_machine_time
    ON predictions (machine_id, predicted_at DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_time
    ON predictions (predicted_at DESC);
