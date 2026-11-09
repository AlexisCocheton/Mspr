-- MECHA — Schéma PostgreSQL pour la simulation production
-- Exécuté automatiquement au premier démarrage du container postgres.
-- L'API applique aussi ce schéma au démarrage (src/database.py:init_schema),
-- ce qui met à niveau une base déjà initialisée par une version antérieure.

CREATE TABLE IF NOT EXISTS sensor_readings (
    id                    SERIAL PRIMARY KEY,
    received_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- heure de réception par l'API
    measured_at           TIMESTAMPTZ,                         -- heure de la mesure à la source
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

-- Alertes générées par les règles de config/regles_metier.yaml (section alertes)
CREATE TABLE IF NOT EXISTS alertes (
    id             SERIAL PRIMARY KEY,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    prediction_id  INTEGER      REFERENCES predictions(id) ON DELETE SET NULL,
    machine_id     VARCHAR(64)  NOT NULL,
    usine_id       VARCHAR(32)  NOT NULL,
    type_machine   VARCHAR(64)  NOT NULL,
    source         VARCHAR(16)  NOT NULL,                     -- panne24h | anomalie
    niveau         VARCHAR(16)  NOT NULL,
    probabilite    REAL,
    rul_hours      REAL,
    message        TEXT         NOT NULL,
    statut         VARCHAR(16)  NOT NULL DEFAULT 'ouverte',   -- ouverte | acquittee
    acquittee_at   TIMESTAMPTZ,
    notifiee       BOOLEAN      NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_alertes_statut_time
    ON alertes (statut, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_alertes_machine_source
    ON alertes (machine_id, source, statut);

-- Exemple de requête d'exploitation : alertes ouvertes par usine et par niveau
-- SELECT usine_id, niveau, COUNT(*) AS nb
-- FROM alertes WHERE statut = 'ouverte'
-- GROUP BY usine_id, niveau ORDER BY usine_id, nb DESC;
