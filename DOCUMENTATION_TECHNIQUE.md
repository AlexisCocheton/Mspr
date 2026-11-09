# Documentation technique — MECHA Maintenance Prédictive

*MSPR TPRE841 — Bloc 4 — EPSI 2025-2026*

Ce document décrit l'architecture, les données, les modèles, la solution applicative et le déploiement du prototype. Tous les chiffres proviennent des artefacts produits par le code : `models/training_results.json` (métriques), `data/processed/rapport_qualite.json` (nettoyage) et `config/regles_metier.yaml` (paramétrage). Ils changent si l'on régénère les données ou ré-entraîne les modèles.

---

## Sommaire

1. [Architecture de la solution](#1-architecture-de-la-solution)
2. [Choix techniques](#2-choix-techniques)
3. [Données](#3-données)
4. [Modèles de machine learning](#4-modèles-de-machine-learning)
5. [Solution applicative intégrée](#5-solution-applicative-intégrée)
6. [Validation de la règle d'alerte](#6-validation-de-la-règle-dalerte)
7. [Déploiement et exploitation](#7-déploiement-et-exploitation)
8. [Enjeux RGPD et protection des données](#8-enjeux-rgpd-et-protection-des-données)

---

## 1. Architecture de la solution

### 1.1 Schéma d'ensemble

```
 SOURCES (atelier)                       FLUX TEMPS RÉEL (1 relevé / heure / machine)
 ─────────────────                       ────────────────────────────────────────────
 ┌──────────────────────┐  HTTP POST JSON  ┌──────────────────────────────────────────┐
 │ Capteurs IoT machines│  /webhook/sensor │ API FastAPI  (src/api.py, port 8000)      │
 │ (simulés : src/      │ ───────────────► │  1. contrôle de plausibilité (bornes YAML)│
 │  sensor_simulator.py)│  + measured_at   │  2. inférence panne 24 h / RUL / anomalie │
 ├──────────────────────┤                  │  3. persistance lecture + prédiction      │
 │ SCADA / MES (cible)  │ ─ ─ ─ ─ ─ ─ ─ ─► │  4. règle d'alerte (N lectures consécut.) │
 │ état machine, arrêts │                  │  5. notification webhook externe          │
 ├──────────────────────┤                  └───────┬───────────────▲───────────┬───────┘
 │ Historique maintenance│                         │ SQL           │ charge    │ POST JSON
 │ (h_depuis_maintenance,│                         ▼               │ .joblib   ▼
 │  nb_maintenances)     │          ┌─────────────────────────┐   │   ┌──────────────────┐
 └──────────┬───────────┘          │ PostgreSQL 15            │   │   │ GMAO / ticketing │
            │                       │  sensor_readings         │   │   │ SCADA, messagerie│
            │ export horaire        │   (measured_at,          │   │   │ (URL configurable)│
            ▼                       │    received_at)          │   │   └──────────────────┘
 ┌──────────────────────┐          │  predictions             │   │
 │ data/raw/            │          │  alertes (ouverte/       │   │
 │ mecha_dataset_full   │          │           acquittee)     │   │
 │ .csv  (BRUT)         │          └────────────▲────────────┘   │
 └──────────┬───────────┘                       │ via API        │
            │                                    │               │
 FLUX BATCH (entraînement)                       │      ┌────────┴─────────┐
 ─────────────────────────                       │      │ models/*.joblib  │
            ▼                                    │      │ training_results │
 ┌──────────────────────┐   ┌──────────────────┐ │      └────────▲─────────┘
 │ data_preparation.py  │──►│ model_training.py│─┼───────────────┘
 │ nettoyage + features │   │ 6 classifieurs,  │ │   (service « trainer » :
 │ → data/processed/    │   │ RUL, Isol. Forest│ │    src/bootstrap.py)
 └──────────────────────┘   └──────────────────┘ │
                                                  │
 ┌──────────────────────────────────────┐        │     ┌───────────────────────────┐
 │ config/regles_metier.yaml            │        └─────│ Dashboard Streamlit       │
 │ bornes capteurs, nettoyage, seuils,  │──────────────│ (src/dashboard.py, 8501)  │
 │ règles d'alerte, fréquences          │  lu par tous │ 5 pages, alertes, SHAP    │
 └──────────────────────────────────────┘  les modules └───────────────────────────┘
```

### 1.2 Flux de données

| Flux | Fréquence | Chemin |
|------|-----------|--------|
| **Temps réel** | 1 relevé par heure et par machine (simulé : 1 heure = 2 s) | capteur → `POST /webhook/sensor` → contrôle des bornes → 3 inférences → `sensor_readings` + `predictions` → règle d'alerte → `alertes` → notification → dashboard (page Monitoring, rafraîchie toutes les 5 s) |
| **Batch** | à l'initialisation, puis ré-entraînement trimestriel | `data/raw/` (brut) → nettoyage + features → `data/processed/` → entraînement → `models/*.joblib` + `training_results.json` |
| **Consultation** | à la demande | dashboard → API (`/predict/*`, `/monitoring/*`, `/alertes`) ; le dashboard ne charge aucun modèle lui-même |

### 1.3 Justification des choix d'architecture

| Contrainte MECHA | Réponse apportée |
|------------------|------------------|
| **5 usines, niveaux d'équipement hétérogènes** | Une stack Docker identique par site (ou centralisée), paramétrée par un fichier de règles. Le chemin du fichier est surchargeable (`MECHA_REGLES`) : un site peut avoir ses propres seuils sans modifier le code. |
| **Données confidentielles, on-premise** | Aucun service cloud : PostgreSQL, API et dashboard tournent sur le serveur du site. Les données ne quittent pas le réseau de l'usine. |
| **Cohérence temporelle (horodatage, synchronisation)** | Chaque lecture porte deux horodatages : `measured_at` (heure de mesure à la source, transmise par le capteur/SCADA) et `received_at` (heure de réception par l'API). Le décalage entre les deux est mesurable, et l'analyse se fait sur l'heure de mesure. |
| **Qualité des données (bruit, trous, capteurs défaillants)** | Double contrôle : à l'ingestion, l'API rejette toute valeur hors bornes physiques (HTTP 422) ; à la préparation, le nettoyage traite doublons, valeurs aberrantes, capteurs figés et trous de mesure (§3.4). |
| **Volumes de données** | 30 machines × 18 mois = 354 162 relevés bruts (45 Mo de CSV). Extrapolé aux ~100 machines critiques citées par la Direction industrielle : ~876 000 relevés par an au pas horaire, volume courant pour PostgreSQL avec les index posés sur `(machine_id, predicted_at)`. Au-delà (pas à la minute, ×60), prévoir le partitionnement par mois ou une extension séries temporelles (TimescaleDB). |
| **Tolérance aux pannes** | L'API démarre et prédit même si PostgreSQL est indisponible : seuls le webhook, le monitoring et les alertes renvoient 503. Une erreur dans l'évaluation des alertes ne fait pas perdre la prédiction enregistrée. Healthchecks Docker sur PostgreSQL et l'API, redémarrage automatique du simulateur. |
| **Démarrage sur un environnement vierge** | Le service `trainer` (`src/bootstrap.py`) prépare les données et entraîne les modèles s'ils sont absents, avant le démarrage de l'API. |
| **Intégration au SI existant** | API REST documentée (OpenAPI/Swagger sur `/docs`), base SQL standard interrogeable par les outils de reporting, notification sortante par webhook JSON vers la GMAO, le ticketing ou le SCADA. |

---

## 2. Choix techniques

### 2.1 Outils et langages

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| Langage | Python 3.11 / 3.12 | Écosystème data science de référence ; une seule langue pour la préparation, les modèles, l'API et le dashboard. |
| Données | pandas, numpy | Manipulation de séries temporelles tabulaires. |
| ML | scikit-learn | Arbres, ensembles, régressions et Isolation Forest dans une API homogène ; modèles sérialisables avec joblib. |
| Explicabilité | SHAP (TreeExplainer) | Contribution de chaque mesure à une prédiction ; exacte et rapide pour les modèles à base d'arbres. |
| API | FastAPI + Pydantic | Validation des entrées déclarative (bornes physiques), documentation Swagger générée automatiquement. |
| Base de données | PostgreSQL 15 + SQLAlchemy Core | Base relationnelle standard, requêtes SQL lisibles (`DISTINCT ON`), sans couche ORM superflue. |
| Paramétrage | YAML (PyYAML) | Format lisible et commentable par un non-développeur ; validé au démarrage. |
| Dashboard | Streamlit + Plotly | Interfaces métier interactives en Python, sans développement front-end. |
| Conteneurisation | Docker + docker compose | Déploiement identique sur chaque site, en une commande. |
| CI | GitHub Actions | Intégré au dépôt Git ; tests et construction de l'image à chaque push. |
| Tests | pytest | 88 tests automatisés (§7.4). |

### 2.2 Structure du code

```
mspr/
├── config/
│   └── regles_metier.yaml      # Paramétrage applicatif (seuils, alertes, fréquences)
├── reference/
│   └── generate_dataset.py     # Générateur du dataset synthétique (brut, avec défauts)
├── data/
│   ├── raw/                    # mecha_dataset_full.csv (brut, versionné)
│   └── processed/              # Dataset nettoyé + features, category_maps.json, rapport_qualite.json
├── src/
│   ├── regles.py               # Lecture et validation des règles métier (point d'accès unique)
│   ├── data_preparation.py     # Nettoyage, features, labels
│   ├── model_training.py       # Entraînement, évaluation, validation de la règle d'alerte
│   ├── bootstrap.py            # Prépare et entraîne si les modèles sont absents (service trainer)
│   ├── api.py                  # API REST FastAPI
│   ├── database.py             # Accès PostgreSQL (lectures, prédictions, alertes)
│   ├── dashboard.py            # Dashboard Streamlit (consomme l'API)
│   └── sensor_simulator.py     # Simulateur du parc IoT (rejoue le CSV brut)
├── db/init.sql                 # Schéma PostgreSQL
├── models/                     # Modèles entraînés (générés, non versionnés)
├── notebooks/
│   └── exploration_mecha.ipynb # Analyse exploratoire du dataset MECHA
├── tests/                      # test_api, test_data_preparation, test_model, test_regles
├── docs/                       # Documentation métier, validation, prompts IA
├── Dockerfile, docker-compose.yml, .dockerignore, .env.example
└── .github/workflows/ci.yml    # Intégration continue
```

Principes de structuration :
- **Une responsabilité par module** ; la logique métier (niveaux de risque, règles d'alerte) est isolée dans `src/regles.py` et utilisée par la préparation, l'entraînement, l'API et le dashboard. Aucun seuil n'est codé en dur ailleurs.
- **Le dashboard ne charge aucun modèle** : il passe par l'API, comme le ferait n'importe quel autre client du SI.
- **Reproductibilité** : graines aléatoires fixées (`SEED = 42` pour la simulation, `random_state=42` pour les modèles).

### 2.3 Gestion du code source

- Dépôt Git hébergé sur GitHub, branche principale `main`.
- Versionnés : code, configuration, schéma SQL, dataset brut (`data/raw/mecha_dataset_full.csv`), mappings d'encodage, bilan qualité, documentation.
- Non versionnés (`.gitignore`) : modèles `.joblib`, graphiques, dataset préparé, environnement virtuel, fichier `.env` (identifiants). Ils sont régénérés par le pipeline.
- Chaque push sur `main` ou `develop` et chaque pull request vers `main` déclenche la CI (§7.4).

---

## 3. Données

### 3.1 Origine

Le dataset est **entièrement synthétique**, généré par `reference/generate_dataset.py` (graine fixe : régénération à l'identique). Ce choix respecte la consigne de confidentialité (aucune donnée réelle d'usine) et permet de contrôler les scénarios de dégradation et de panne. Les datasets publics étudiés au début du projet (AI4I 2020, Azure Predictive Maintenance) ont été abandonnés au profit de ce dataset, aligné sur le vocabulaire et l'organisation de MECHA.

| Paramètre | Valeur |
|-----------|--------|
| Période | Sites français : 01/01/2024 → 29/06/2025 (13 104 h par machine). Sites espagnols : 01/07/2024 → 29/06/2025 (8 736 h par machine) |
| Granularité | Horaire |
| Usines | 5 : USN-FR-01 Toulouse (8 machines), USN-FR-02 Lyon (6), USN-FR-03 Bordeaux (7), USN-ES-01 Barcelone (5), USN-ES-02 Valence (4) |
| Machines | 30, de 4 types : CNC-Fraisage, CNC-Tournage, Découpe-Laser, Centre-Usinage |
| Volume brut | 354 162 lignes × 17 colonnes (dont 354 doublons de transmission) |
| Volume après nettoyage | 351 605 lignes |
| Taux de relevés en panne | 2,45 % |
| Taux de relevés « panne dans les 24 h » | 2,32 % |
| Pannes par machine | 9 à 17 (sites français), 7 à 11 (sites espagnols, historique plus court) |

### 3.2 Hypothèses de simulation

**Cycle de vie des machines**
- Âge initial tiré entre 1 000 et 50 000 h ; heures depuis la dernière maintenance au départ : 0 à 600 h.
- **MTBF dépendant de l'âge** : `MTBF = 1440 / (1 + âge / 50 000 × 0,8)`, soit de 1 440 h (machine neuve) à 800 h (machine de 50 000 h). L'intervalle entre deux pannes suit une loi normale centrée sur le MTBF courant (écart-type 20 %), **avec un minimum de 300 h**.
- Durée d'une panne : 4 à 48 h. À la fin d'une panne, une maintenance corrective remet `h_depuis_maintenance` à 0.
- **RUL** (`rul_heures`) : heures restantes avant le début de la prochaine panne, plafonné à 720 h.
- **États** : `en_panne` pendant une panne ; `critique` si RUL < 48 h ; `a_risque` si RUL < 168 h ; `normal` sinon.

**Comportement des capteurs**
- Score de dégradation : `1 − RUL/720`, augmenté d'une composante « absence de maintenance » (`h_depuis_maintenance / 4000`, plafonnée à 0,25), plus un bruit gaussien (σ = 0,03).
- Facteur appliqué aux valeurs nominales : `1 + score × 0,30 + dérive`, où la dérive due au manque de maintenance vaut `h_depuis_maintenance / 2000` (plafonnée à 0,30). Facteur majoré de 7 % dans les 48 h qui précèdent une panne.
- Valeurs nominales par type : Fraisage 65 °C / 2,1 mm/s / 18 A ; Tournage 60 °C / 1,8 mm/s / 15 A ; Laser 45 °C / 0,8 mm/s / 22 A ; Centre d'usinage 70 °C / 2,5 mm/s / 25 A. Pression nominale 6,5 bar, vitesse nominale 1 500 tr/min.
- Température, vibration et courant augmentent avec la dégradation ; pression et vitesse diminuent.
- Bruit industriel : bruit multiplicatif ±4 %, bruit additif (température σ = 10 °C, vibration σ = 0,55 mm/s, courant σ = 3,5 A, pression σ = 0,45 bar, vitesse σ = 90 tr/min), pics ponctuels (1 % des relevés).
- Pendant une panne : valeurs dégradées et plus bruitées, volontairement proches de l'état « critique » pour éviter un problème trivial.
- **Saisonnalité** : en été, +10 °C ambiants en climat méditerranéen, +6 °C en climat chaud, +3 °C en climat tempéré ; −3 °C en hiver.

**Hétérogénéité des sites et défauts de qualité** (fichier brut)

Les sites espagnols, équipés plus récemment, ont 6 mois d'historique de moins et des capteurs plus sujets aux pertes de mesure. Les défauts sont injectés après la simulation par un générateur aléatoire distinct (la simulation physique n'est pas modifiée) ; les labels ne sont jamais altérés.

| Défaut injecté | Taux / fréquence |
|----------------|------------------|
| Valeur manquante ponctuelle | 0,5 % des mesures (France), 1,2 % (Espagne), par capteur |
| Perte de transmission (tous capteurs absents) | ~6 épisodes par machine et par an, de 1 à 6 h |
| Capteur figé (valeur répétée) | ~3 épisodes par machine et par an, de 6 à 24 h |
| Valeur aberrante (code d'erreur hors limites physiques, ex. 999,9 °C, 65 535 tr/min) | 0,05 % des mesures, par capteur |
| Doublon de transmission | 0,1 % des relevés |

### 3.3 Dictionnaire de données — `data/raw/mecha_dataset_full.csv`

Les domaines de valeurs indiqués sont ceux des données **nettoyées** ; le fichier brut contient en plus les valeurs aberrantes et manquantes décrites ci-dessus.

| Colonne | Type | Unité | Domaine de valeurs | Description |
|---------|------|-------|--------------------|-------------|
| `timestamp` | datetime | — | 2024-01-01 00:00 → 2025-06-29 23:00, pas horaire | Horodatage du relevé |
| `usine_id` | catégoriel | — | USN-FR-01, USN-FR-02, USN-FR-03, USN-ES-01, USN-ES-02 | Usine |
| `machine_id` | catégoriel | — | format `USN-FR-01-M01` (usine + numéro de machine) | Identifiant unique de la machine |
| `type_machine` | catégoriel | — | CNC-Fraisage, CNC-Tournage, Découpe-Laser, Centre-Usinage | Type de machine |
| `age_machine_h` | entier | h | ~2 600 à ~63 000 | Âge depuis la mise en service |
| `h_depuis_maintenance` | entier | h | 1 à ~1 800 | Heures depuis la dernière intervention |
| `temperature_C` | réel | °C | 6 à 176 (moyenne 81) ; bornes de plausibilité −20 / 300 | Température machine |
| `vibration_mm_s` | réel | mm/s | 0 à 8,2 (moyenne 2,5) ; bornes 0 / 20 | Vibration |
| `courant_A` | réel | A | 0,9 à 65 (moyenne 25) ; bornes 0 / 100 | Courant absorbé |
| `pression_bar` | réel | bar | 2,5 à 8,4 (moyenne 6,0) ; bornes 0 / 20 | Pression hydraulique |
| `vitesse_tr_min` | réel | tr/min | 529 à 1 864 (moyenne 1 351) ; bornes 0 / 3 000 | Vitesse de broche |
| `score_degradation` | réel | — | 0 à 1 | Score interne de dégradation (variable de simulation, non utilisée comme feature) |
| `rul_heures` | entier | h | 1 à 720 (plafonné) | Heures avant le début de la prochaine panne |
| `etat_machine` | catégoriel | — | normal, a_risque, critique, en_panne | État qualitatif dérivé du RUL |
| `en_panne` | booléen (0/1) | — | 0 / 1 | Panne en cours |
| `nb_pannes_total` | entier | — | 0 à 17 | Pannes cumulées sur la période |
| `nb_maintenances_total` | entier | — | 0 à 17 | Maintenances correctives cumulées |

Colonnes ajoutées par `src/data_preparation.py` dans `data/processed/mecha_unified_prepared.csv` :

| Colonne | Type | Description |
|---------|------|-------------|
| `<capteur>_mean_24h`, `<capteur>_std_24h` (10 colonnes) | réel | Moyenne et écart-type glissants sur les 24 derniers relevés de la machine |
| `type_machine_encoded`, `usine_encoded` | entier | Encodage des catégories (table dans `category_maps.json`) |
| `panne_dans_24h` | 0/1 | **Cible prédictive** : la prochaine panne commence dans les 24 h (`rul_heures ≤ 24`). Pendant une panne, le RUL pointe vers la panne suivante, donc ces relevés valent 0. |

### 3.4 Qualité des données et nettoyage

`src/data_preparation.py` applique, dans l'ordre, les règles de la section `qualite_donnees` du fichier de règles :

| Étape | Règle | Résultat |
|-------|-------|----------|
| 1. Doublons | Même machine et même horodatage : on garde le premier | 354 relevés supprimés |
| 2. Valeurs manquantes initiales | — | 15 553 valeurs |
| 3. Valeurs hors bornes physiques | Remplacées par une valeur manquante (bornes de la section `capteurs`) | 829 valeurs |
| 4. Capteur figé | Même valeur pendant 6 h ou plus : valeur manquante, sauf la première mesure de l'épisode | 1 610 valeurs |
| 5. Imputation | Trou de 3 h au plus comblé par la dernière valeur connue de la machine (ce dont dispose aussi l'atelier en temps réel ; pas d'interpolation avec le futur) | 13 574 valeurs comblées |
| 6. Lignes incomplètes | Trous plus longs : ligne écartée de l'apprentissage | 2 203 lignes (0,6 %) |

Bilan : 354 162 lignes brutes → **351 605 lignes exploitables**. Le détail est sauvegardé dans `data/processed/rapport_qualite.json` et affiché dans le dashboard.

### 3.5 Limites connues des données

- Données simulées : les relations capteurs ↔ dégradation sont quasi linéaires et un seul mode de défaillance est modélisé (pas de pannes multi-causes).
- Pas de journal de maintenance détaillé (pièces changées, maintenances préventives) : l'historique de maintenance est résumé par `h_depuis_maintenance` et `nb_maintenances_total`, et seules des maintenances correctives sont simulées.
- Pas de variable « temps de cycle » ni de codes défaut machine.
- 30 machines simulées contre ~100 machines critiques dans le parc réel.
- Les modèles entraînés sur ces données devront être ré-entraînés sur les données réelles du site pilote.

---

## 4. Modèles de machine learning

### 4.1 Tâches et features

| Tâche | Cible | Modèle déployé |
|-------|-------|----------------|
| 1. Machine en panne | `en_panne` | Random Forest (`/predict/panne`) |
| 2. **Panne dans les 24 h** (prédictif, utilisé pour les alertes) | `panne_dans_24h` | Random Forest (`/predict/panne24h`) |
| 3. Temps restant avant panne | `rul_heures` (relevés hors panne) | Random Forest (`/predict/rul`) |
| 4. Comportement atypique | non supervisé | Isolation Forest (`/predict/anomalie`) |

**Features (9)** : les 5 mesures instantanées (`temperature_C`, `vibration_mm_s`, `courant_A`, `pression_bar`, `vitesse_tr_min`), `age_machine_h`, `h_depuis_maintenance`, `type_machine_encoded`, `usine_encoded`. Elles sont standardisées (`StandardScaler`, ajusté sur le train uniquement).

Les moyennes et écarts-types glissants sur 24 h sont calculés mais **exclus de l'entraînement**, par choix de conception : le modèle travaille sur le relevé courant, seul disponible sans historique. C'est aussi sa principale limite (§4.7).

### 4.2 Protocole d'évaluation

**Classification (tâches 1 et 2)**
- Découpage aléatoire stratifié 80 / 20. Il n'y a aucun relevé commun entre train et test (vérifié).
- Train rééquilibré 50/50 par sous-échantillonnage des relevés sans panne.
- **Deux évaluations complémentaires** :
  - **Test rééquilibré 50/50**, à la demande de l'encadrant : il vérifie que le modèle reconnaît réellement une panne. Sur les données réelles (2,3 % de pannes), un modèle qui répond toujours « pas de panne » obtient 97,7 % d'accuracy sans rien détecter ; sur un test 50/50, il plafonne à 50 %. C'est le jeu qui sert à comparer les modèles.
  - **Test à la distribution réelle** (70 321 relevés, non rééquilibré) : il mesure ce que verra la maintenance, notamment la part de fausses alertes par niveau de risque.
- Un **modèle naïf** (répond toujours « pas de panne ») sert de référence.
- Validation croisée 5-fold sur le train pour Random Forest, Gradient Boosting et arbre de décision.
- Échantillons : 13 038 relevés en train et 3 260 en test 50/50 pour la tâche 24 h ; 13 790 et 3 448 pour la tâche « en panne ».

**Régression RUL (tâche 3)**
- **Découpage temporel** : train avant le 23/03/2025, test après (274 368 / 68 618 relevés, hors relevés en panne). Un découpage aléatoire mettrait des heures voisines d'une même machine dans le train et le test, ce qui surestime fortement la performance : lors d'un essai sur la version précédente du dataset, le R² du Random Forest passait de 0,87 en découpage aléatoire à 0,65 en découpage temporel.
- Références : modèle naïf (RUL moyen du train) et régression linéaire.

### 4.3 Résultats — panne dans les 24 h

| Modèle | Accuracy | Précision | Rappel | F1 | AUC-ROC | F1 CV 5-fold |
|--------|----------|-----------|--------|----|---------|--------------|
| Modèle naïf | 0,500 | 0,000 | 0,000 | 0,000 | 0,500 | — |
| **Random Forest** (déployé) | 0,874 | 0,830 | 0,940 | 0,882 | 0,940 | 0,881 ± 0,008 |
| Gradient Boosting | 0,900 | 0,865 | 0,949 | 0,905 | 0,959 | 0,899 ± 0,011 |
| Arbre de décision | 0,836 | 0,810 | 0,877 | 0,842 | 0,870 | 0,839 ± 0,005 |
| Régression logistique | 0,848 | 0,817 | 0,896 | 0,855 | 0,912 | — |
| KNN | 0,849 | 0,813 | 0,907 | 0,858 | 0,905 | — |
| SVM | 0,864 | 0,813 | 0,945 | 0,874 | 0,932 | — |

*Test 50/50.*

| Modèle | Accuracy | Précision | Rappel | F1 | Fausses alertes « critique » | Fausses alertes « élevé » |
|--------|----------|-----------|--------|----|------------------------------|---------------------------|
| Modèle naïf | 0,977 | 0,000 | 0,000 | 0,000 | — | — |
| **Random Forest** (déployé) | 0,824 | 0,111 | 0,940 | 0,198 | 79,3 % | 94,3 % |
| Gradient Boosting | 0,862 | 0,138 | 0,949 | 0,242 | 76,5 % | 95,2 % |
| Arbre de décision | 0,810 | 0,098 | 0,877 | 0,176 | 88,1 % | 95,4 % |
| Régression logistique | 0,812 | 0,101 | 0,896 | 0,181 | 85,1 % | 93,4 % |
| KNN | 0,810 | 0,101 | 0,907 | 0,181 | 86,4 % | 94,1 % |
| SVM | 0,801 | 0,100 | 0,945 | 0,180 | 82,9 % | 95,3 % |

*Test à la distribution réelle (2,32 % de positifs), comptage relevé par relevé.*

**Lecture** : les modèles reconnaissent bien une machine proche de la panne (rappel 94 % pour le Random Forest, AUC 0,94). En revanche, relevé par relevé et au taux de pannes réel, 79 % des prédictions « critique » du Random Forest ne sont pas suivies d'une panne dans les 24 h. C'est pourquoi la solution ne déclenche pas d'alerte sur un relevé isolé, mais sur une règle validée par événement (§6).

### 4.4 Résultats — machine en panne

| Modèle | F1 (test 50/50) | AUC-ROC | F1 CV 5-fold | Précision (réel) | Rappel (réel) | Fausses alertes « critique » (réel) |
|--------|-----------------|---------|--------------|------------------|---------------|-------------------------------------|
| Modèle naïf | 0,000 | 0,500 | — | 0,000 | 0,000 | — |
| **Random Forest** (déployé) | 0,922 | 0,977 | 0,921 ± 0,002 | 0,194 | 0,945 | 61,5 % |
| Gradient Boosting | 0,935 | 0,986 | 0,936 ± 0,002 | 0,231 | 0,952 | 60,3 % |
| Arbre de décision | 0,889 | 0,941 | 0,890 ± 0,006 | 0,180 | 0,885 | 76,2 % |
| Régression logistique | 0,888 | 0,960 | — | 0,153 | 0,903 | 70,6 % |
| KNN | 0,904 | 0,964 | — | 0,219 | 0,897 | 63,4 % |
| SVM | 0,925 | 0,979 | — | 0,202 | 0,942 | 63,5 % |

En production, l'état « en panne » est en général connu du MES (machine arrêtée) : ce modèle sert surtout de contrôle de cohérence et d'appui à l'explication (`/predict/panne/explain`).

### 4.5 Résultats — temps restant avant panne (RUL)

| Modèle | MAE | RMSE | R² |
|--------|-----|------|----|
| Modèle naïf (moyenne) | 223,7 h | 248,5 h | 0,000 |
| Régression linéaire | 105,3 h | 134,4 h | 0,708 |
| **Random Forest** (déployé) | 105,5 h | 155,8 h | 0,607 |
| Gradient Boosting | 101,5 h | 147,3 h | 0,649 |

*Découpage temporel.*

**Lecture** : tous les modèles divisent par deux l'erreur du modèle naïf, mais **la régression linéaire fait aussi bien que les arbres**. Dans les données, le RUL est surtout piloté par `h_depuis_maintenance` (le compteur repart de zéro après chaque panne), relation que le modèle linéaire capte directement. Avec une erreur moyenne d'environ 100 h, le RUL est un **ordre de grandeur utile pour prioriser** les machines, pas une échéance : il n'est pas utilisé pour déclencher les alertes.

### 4.6 Détection d'anomalies (Isolation Forest)

Entraîné sans label sur les 286 220 relevés à l'état normal (`n_estimators = 200`, `contamination = 0,05`). Il signale 83,1 % des relevés en panne comme anormaux, pour 5,0 % de faux positifs sur les relevés normaux (taux fixé par `contamination`). Son rôle est de détecter un comportement capteur inhabituel que les modèles supervisés n'ont jamais vu (nouveau mode de défaillance, capteur déréglé).

### 4.7 Hyperparamètres retenus et choix du modèle déployé

| Modèle | Paramètres principaux |
|--------|-----------------------|
| Random Forest (classification) | `n_estimators=200`, `max_depth=15`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features="sqrt"`, `class_weight="balanced"`, `random_state=42` |
| Gradient Boosting (classification) | `n_estimators=200`, `max_depth=5`, `learning_rate=0.1`, `min_samples_split=5`, `subsample=1.0` |
| Arbre de décision | `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=2`, `class_weight="balanced"` |
| Régression logistique | `C=1.0`, `solver="lbfgs"`, `max_iter=1000`, `class_weight="balanced"` |
| KNN | `n_neighbors=5`, `weights="distance"`, distance euclidienne |
| SVM | noyau RBF, `C=1.0`, `gamma="scale"`, `class_weight="balanced"`, `probability=True` ; entraîné sur 10 000 relevés tirés du train |
| Random Forest (RUL) | `n_estimators=200`, `max_depth=15`, `min_samples_split=5`, `max_features=1.0` |
| Gradient Boosting (RUL) | `n_estimators=200`, `max_depth=5`, `learning_rate=0.1`, `min_samples_split=5` |
| Isolation Forest | `n_estimators=200`, `contamination=0.05` |

La liste complète des paramètres de chaque modèle est enregistrée dans `training_results.json` (clé `hyperparametres`). Les paramètres ont été fixés à des valeurs usuelles et n'ont pas fait l'objet d'une recherche systématique (grid search). La faible variance en validation croisée (± 0,002 à 0,011) indique des résultats stables.

**Pourquoi Random Forest dans l'API alors que Gradient Boosting est légèrement meilleur ?** L'écart est faible (F1 50/50 de 0,882 contre 0,905 sur la tâche 24 h, et une part de fausses alertes « critique » proche : 79 % contre 77 %). Le Random Forest l'emporte sur les critères d'exploitation :
- moins sensible au réglage des hyperparamètres (pas de taux d'apprentissage à ajuster site par site) ;
- entraînement parallélisé sur tous les cœurs (`n_jobs=-1`), utile pour des ré-entraînements fréquents ;
- explicable par SHAP TreeExplainer et importances de variables ;
- robuste au bruit capteur grâce à l'agrégation de nombreux arbres.

Le passage au Gradient Boosting reste une piste d'amélioration directe : il suffit de changer le fichier chargé par `load_models()`.

### 4.8 Limites des modèles

- **Features instantanées uniquement** : avec un seul relevé, le modèle reconnaît une machine qui se dégrade, mais pas le moment exact de la panne. Une partie des alertes arrive donc plusieurs jours avant la panne (§6). **Le principal levier d'amélioration** consiste à calculer des features temporelles (tendances, moyennes glissantes, nombre de dépassements sur 24 h) à partir de l'historique déjà stocké dans `sensor_readings`.
- Probabilités issues d'un entraînement 50/50 : elles sont plus élevées que la fréquence réelle des pannes et ne doivent pas être lues comme des probabilités calibrées.
- Modèles entraînés sur des données simulées : ré-entraînement obligatoire sur les données réelles du site pilote.
- Dérive possible (nouvelles machines, changement de procédé) : ré-entraînement trimestriel prévu.

---

## 5. Solution applicative intégrée

### 5.1 Mécanismes d'intégration

| Mécanisme | Usage | Détail |
|-----------|-------|--------|
| **API REST** (JSON) | Ingestion des relevés, prédictions à la demande, consultation | 15 endpoints, documentation Swagger sur `http://<serveur>:8000/docs` |
| **Base PostgreSQL** | Historisation des relevés, prédictions et alertes ; reporting | Tables `sensor_readings`, `predictions`, `alertes` ; interrogeable par tout outil SQL |
| **Fichier de règles YAML** | Paramétrage métier sans modification du code | `config/regles_metier.yaml`, validé au démarrage de l'API |
| **Webhook sortant** | Notification des nouvelles alertes vers la GMAO, un outil de ticketing, le SCADA ou une messagerie | URL dans le YAML ou variable `MECHA_ALERT_WEBHOOK_URL` |
| **Fichiers intermédiaires** | Échanges batch | Import CSV dans le dashboard (endpoint `/predict/batch`), export du rapport des machines à risque |

### 5.2 Paramétrage applicatif — `config/regles_metier.yaml`

Le fichier est lu par la préparation des données, l'entraînement, l'API, le dashboard et le simulateur. Toute incohérence (seuils non décroissants, niveau inconnu, section manquante) bloque le démarrage de l'API avec un message explicite, et la CI vérifie sa validité à chaque push.

```yaml
# Règles de calcul : probabilité de panne dans les 24 h -> niveau de risque
niveaux_risque_panne:
  - {niveau: critique, seuil_min: 0.80, couleur: "#dc3545",
     action: "ARRÊT IMMÉDIAT recommandé. Intervention urgente requise."}
  - {niveau: élevé,    seuil_min: 0.50, action: "Planifier une maintenance préventive ..."}
  - {niveau: moyen,    seuil_min: 0.30, action: "Surveillance renforcée ..."}
  - {niveau: faible,   seuil_min: 0.0,  action: "Machine en fonctionnement normal ..."}

# Temps restant avant défaillance : ordre de grandeur, ne déclenche pas d'alerte
niveaux_risque_rul:
  - {niveau: critique, rul_max_h: 6}
  - {niveau: élevé,    rul_max_h: 12}
  - {niveau: moyen,    rul_max_h: 24}
  - {niveau: faible,   rul_max_h: null}

# Déclenchement des alertes : N lectures consécutives dans un niveau déclencheur.
# Tant qu'une alerte est ouverte (machine + source), aucune nouvelle alerte :
# elle doit d'abord être acquittée par la maintenance.
alertes:
  panne24h: {niveaux: [critique], lectures_consecutives: 3}
  anomalie: {niveaux: [anomalie], lectures_consecutives: 3}
  notification: {webhook_url: "", timeout_s: 3}

# Fréquences de mise à jour
frequences:
  releve_capteurs: "1 h"
  simulateur_intervalle_s: 2.0
  dashboard_rafraichissement_s: 5
  reentrainement_modeles: "trimestriel"
```

Le fichier contient aussi les **bornes physiques des capteurs** (contrôle de plausibilité à l'ingestion et au nettoyage), les **règles de qualité des données** (6 h pour un capteur figé, 3 h d'imputation maximum) et la **normalisation du score d'anomalie** (`score = (−score_brut − 0,3) / 0,4`, borné à [0, 1] ; au-delà de 0,7, une anomalie détectée passe de « suspect » à « anomalie »).

**Règle d'alerte retenue** : « critique » (probabilité ≥ 0,80) sur 3 relevés consécutifs. C'est le meilleur compromis mesuré lors de la validation (§6) : un volume d'alertes supportable (0,53 alerte par machine et par semaine) pour 75 % des pannes anticipées.

### 5.3 Formalismes de configuration et d'exploitation

**Requêtes SQL** (`src/database.py`, `db/init.sql`) — dernier état connu de chaque machine, en un seul parcours grâce à `DISTINCT ON` (PostgreSQL) :

```sql
SELECT DISTINCT ON (machine_id)
    machine_id, usine_id, type_machine, predicted_at,
    panne24h_probability, panne24h_risk_level,
    rul_hours, rul_risk_level, is_anomaly, anomaly_risk_level
FROM predictions
ORDER BY machine_id, predicted_at DESC;
```

Exemple de requête d'exploitation pour le pilotage industriel (alertes ouvertes par usine et par niveau) :

```sql
SELECT usine_id, niveau, COUNT(*) AS nb
FROM alertes WHERE statut = 'ouverte'
GROUP BY usine_id, niveau ORDER BY usine_id, nb DESC;
```

**Format d'échange JSON** — relevé envoyé par un capteur ou le SCADA (`POST /webhook/sensor`) :

```json
{
  "machine_id": "USN-FR-01-M03",
  "usine_id": "USN-FR-01",
  "type_machine": "CNC-Fraisage",
  "measured_at": "2025-06-01T08:00:00",
  "temperature_C": 112.0,
  "vibration_mm_s": 3.9,
  "courant_A": 33.0,
  "pression_bar": 5.0,
  "vitesse_tr_min": 1120,
  "age_machine_h": 30000,
  "h_depuis_maintenance": 1100
}
```

Réponse de l'API (valeurs calculées par les modèles actuels). Ici, la troisième lecture « critique » consécutive déclenche une alerte :

```json
{
  "reading_id": 1542,
  "prediction_id": 1542,
  "machine_id": "USN-FR-01-M03",
  "panne24h": {"prediction": 1, "probability": 0.9511, "risk_level": "critique",
               "recommendation": "ARRÊT IMMÉDIAT recommandé. Intervention urgente requise."},
  "rul":      {"estimated_rul_hours": 44.75, "risk_level": "faible",
               "recommendation": "Pas de défaillance estimée à court terme (estimation indicative : se fier au risque de panne 24 h et aux alertes)."},
  "anomaly":  {"is_anomaly": true, "anomaly_score": 0.6902, "risk_level": "suspect",
               "recommendation": "Comportement légèrement atypique. Surveillance renforcée recommandée."},
  "alertes": [
    {"id": 12, "machine_id": "USN-FR-01-M03", "source": "panne24h", "niveau": "critique",
     "message": "Niveau « critique » sur 3 lectures consécutives. ARRÊT IMMÉDIAT recommandé. Intervention urgente requise.",
     "notifiee": false}
  ]
}
```

L'exemple montre aussi que le niveau RUL (« faible » à 44 h) peut diverger du niveau panne 24 h : le RUL est un ordre de grandeur (§4.5) et seule la probabilité de panne déclenche l'alerte. Les libellés RUL du fichier de règles le rappellent explicitement à l'utilisateur (ils renvoient au risque de panne 24 h au lieu d'affirmer que la machine est saine).

Le webhook de notification reçoit la ligne complète de l'alerte (`id`, `created_at`, `machine_id`, `usine_id`, `type_machine`, `source`, `niveau`, `probabilite`, `rul_hours`, `message`, `statut`…) en `POST` JSON.

### 5.4 Endpoints de l'API

| Méthode | Endpoint | Rôle |
|---------|----------|------|
| GET | `/health` | État de l'API, modèles chargés, disponibilité de la base |
| GET | `/model/info` | Modèles chargés et métriques d'entraînement |
| GET | `/config/regles` | Règles métier actives |
| POST | `/predict/panne` | Machine actuellement en panne (probabilité, niveau, recommandation) |
| POST | `/predict/panne24h` | Panne dans les 24 h |
| POST | `/predict/rul` | Temps restant avant défaillance |
| POST | `/predict/anomalie` | Comportement capteur atypique (Isolation Forest) |
| POST | `/predict/batch` | Panne 24 h + RUL pour 1 à 1 000 relevés (import CSV) |
| POST | `/predict/panne/explain` | 3 mesures qui ont le plus pesé dans la prédiction « en panne » (SHAP) |
| POST | `/predict/panne24h/explain` | Idem pour la prédiction à 24 h |
| POST | `/webhook/sensor` | Ingestion d'un relevé : inférences, persistance, alertes, notification |
| GET | `/monitoring/recent` | Dernières prédictions (200 au maximum) |
| GET | `/monitoring/machines` | Dernière prédiction de chaque machine |
| GET | `/alertes` | Alertes, filtrables par statut (`ouverte` / `acquittee`) |
| POST | `/alertes/{id}/acquitter` | Prise en charge d'une alerte par la maintenance |

Toute requête dont une mesure sort des bornes physiques du fichier de règles est rejetée (HTTP 422). Si les modèles ne sont pas chargés, les prédictions renvoient 503 ; sans base de données, le webhook, le monitoring et les alertes renvoient 503, mais les prédictions restent disponibles.

### 5.5 Écrans du dashboard et usages

| Page | Contenu | Utilisateurs | Usage attendu |
|------|---------|--------------|---------------|
| **Vue d'ensemble** | Usines, machines, taux de panne ; performance de la règle d'alerte (pannes anticipées, alertes suivies d'une panne, alertes par machine et par semaine) ; bilan qualité des données ; états par usine ; machines les plus touchées ; évolution des pannes | Direction industrielle, pilotage | Suivi hebdomadaire de la performance du parc et de la solution |
| **Prédiction en temps réel** | Saisie manuelle d'un relevé : probabilité, jauge, niveau, recommandation, RUL, explication SHAP ; import CSV d'un lot de relevés avec rapport téléchargeable des machines à risque | Techniciens, responsables maintenance | Vérifier une machine suspecte ; analyser un lot de relevés (export SCADA) |
| **Analyse des capteurs** | Courbes des 500 derniers relevés d'une machine avec les périodes de panne, corrélations, distribution par état | Ingénieurs fiabilité | Confirmer une alerte en regardant la tendance des capteurs |
| **Performance des modèles** | Les deux évaluations (50/50 et distribution réelle), RUL, Isolation Forest, tableau de validation de la règle d'alerte, graphiques (matrices de confusion, courbes précision-rappel, importances) | DSI, équipe data | Contrôler la qualité des modèles après chaque ré-entraînement |
| **Monitoring temps réel** | Alertes ouvertes avec bouton d'acquittement, indicateurs, jauge par machine filtrable par usine, dernières prédictions ; rafraîchissement toutes les 5 s | Responsables maintenance, chefs d'atelier | Écran de supervision : traiter les alertes, prioriser les interventions |

---

## 6. Validation de la règle d'alerte

L'évaluation relevé par relevé (§4.3) ne dit pas combien d'alertes recevra la maintenance. La règle d'alerte est donc validée **par événement**, dans des conditions proches de la production (`valider_regle_alerte` dans `src/model_training.py`) :
- Random Forest panne 24 h entraîné sur les relevés antérieurs au 23/03/2025, puis rejoué heure par heure sur la période suivante (30 machines, 419 machine-semaines, 77 pannes).
- Une alerte est justifiée si une panne commence dans les 24 h ; deux alertes d'une même machine sont séparées d'au moins 24 h (approximation de « alerte ouverte jusqu'à acquittement ») ; les heures où la machine est déjà en panne sont ignorées (arrêt connu du MES).

| Règle | Alertes | Pannes anticipées | Alertes suivies d'une panne sous 24 h | sous 72 h | sous 7 j | Alertes / machine / semaine |
|-------|---------|-------------------|---------------------------------------|-----------|----------|-----------------------------|
| ≥ 0,50 sur 1 relevé | 1 012 | 97,4 % | 7,5 % | 22,2 % | 48,4 % | 2,42 |
| ≥ 0,80 sur 1 relevé | 561 | 93,5 % | 12,9 % | 35,9 % | 67,8 % | 1,34 |
| ≥ 0,80 sur 2 relevés | 324 | 81,8 % | 19,4 % | 53,3 % | 85,8 % | 0,77 |
| **≥ 0,80 sur 3 relevés (configurée)** | **223** | **75,3 %** | **26,0 %** | **61,3 %** | **90,3 %** | **0,53** |
| ≥ 0,80 sur 4 relevés | 162 | 62,3 % | 29,6 % | 68,5 % | 93,6 % | 0,39 |

Avec la règle configurée, **9 alertes sur 10 sont suivies d'une panne dans la semaine**, et 3 pannes sur 4 sont signalées dans les 24 h qui précèdent (anticipation médiane : 14 h). En revanche, seules 26 % des alertes sont suivies d'une panne dans les 24 h : l'alerte est fiable sur la machine concernée, moins sur le moment de la panne. Le besoin BM2 du client (« fausses alertes critiques < 5 % ») n'est pas atteint à l'horizon de 24 h. Il est approché à l'horizon d'une semaine (9,7 % d'alertes non suivies d'une panne). La formulation de l'alerte (« intervention à planifier sous 7 jours ») est à valider avec la Direction industrielle.

L'analyse complète (validation technique et fonctionnelle, projection de déploiement) est dans [docs/RAPPORT_VALIDATION_DEPLOIEMENT.md](docs/RAPPORT_VALIDATION_DEPLOIEMENT.md).

---

## 7. Déploiement et exploitation

### 7.1 Docker compose

```bash
cp .env.example .env          # optionnel : identifiants PostgreSQL, URL de notification
docker compose up --build
```

| Service | Rôle | Démarre après |
|---------|------|---------------|
| `postgres` | Base PostgreSQL 15, schéma initialisé par `db/init.sql` | — |
| `trainer` | `python -m src.bootstrap` : prépare les données et entraîne les modèles s'ils sont absents, puis s'arrête | — |
| `api` | API FastAPI (port 8000), applique aussi le schéma SQL au démarrage (mise à niveau d'une base existante) | `postgres` en bonne santé **et** `trainer` terminé avec succès |
| `dashboard` | Streamlit (port 8501) | `api` en bonne santé |
| `simulator` | Rejoue le CSV brut vers le webhook (relevés incomplets non transmis, valeurs aberrantes rejetées par l'API) | `api` en bonne santé |

- **Premier démarrage** sur un clone neuf : le `trainer` entraîne les modèles (10 à 15 minutes) ; les démarrages suivants réutilisent `./models`.
- **Volumes** : `./models` et `./data` sont montés dans les conteneurs (ils ne sont pas dans l'image) ; `./config` est monté **en lecture seule**. Après modification des règles : `docker compose restart api dashboard`.
- **Identifiants** : `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` et `MECHA_ALERT_WEBHOOK_URL` viennent du fichier `.env` (modèle fourni : `.env.example`) ; valeurs par défaut de développement : `mecha`.
- **Image** (`Dockerfile`) : `python:3.12-slim`, dépendances de `requirements.txt`, puis `config/`, `src/` et le générateur. Le `.dockerignore` exclut données, modèles, environnement virtuel et documentation.
- Accès : API `http://localhost:8000` (Swagger : `/docs`), dashboard `http://localhost:8501`.

### 7.2 Exécution locale (sans Docker)

```bash
python -m venv venv
venv\Scripts\activate                 # Windows  (source venv/bin/activate sous Linux/Mac)
pip install -r requirements.txt

python reference/generate_dataset.py  # facultatif : le dataset brut est versionné ; écrit data/raw/mecha_dataset_full.csv
python src/data_preparation.py        # nettoyage + features -> data/processed/
python src/model_training.py          # entraînement et évaluations (10 à 15 min) -> models/
#   ou, pour les deux étapes : python -m src.bootstrap

uvicorn src.api:app --host 0.0.0.0 --port 8000      # API
streamlit run src/dashboard.py                      # dashboard (autre terminal)

# Monitoring temps réel (facultatif) : PostgreSQL + simulateur
docker compose up -d postgres
set API_URL=http://localhost:8000     # export API_URL=... sous Linux/Mac
python src/sensor_simulator.py

pytest tests/ -v                      # tests
```

### 7.3 Prérequis

- **Serveur** : Linux recommandé (ou Windows avec Docker Desktop / WSL2), Docker Engine 20.10+ avec le plugin `docker compose`.
- **Ressources** : 4 cœurs et 8 Go de RAM recommandés. Le ré-entraînement est la phase la plus coûteuse ; les modèles Random Forest occupent plusieurs centaines de Mo sur disque.
- **Réseau** : ports 8000 (API) et 8501 (dashboard) ouverts sur le réseau du site ; PostgreSQL (5432) à ne pas exposer en dehors du serveur en production ; flux des capteurs/SCADA vers l'API en HTTP (ou via une passerelle MQTT → HTTP).
- **Sécurité** : l'API du prototype n'a pas d'authentification. Avant la production : TLS, authentification (jeton ou SSO de l'entreprise) et filtrage réseau.

### 7.4 Intégration continue (`.github/workflows/ci.yml`)

| Job | Étapes |
|-----|--------|
| `test` (Python 3.11 et 3.12) | Installation des dépendances (cache pip) → validation de `config/regles_metier.yaml` → `pytest tests/` |
| `build-docker` (si les tests passent) | Construction de l'image → test de fumée dans le conteneur (import de l'API et du bootstrap, couche base de données importable) → validation de `docker-compose.yml` (`docker compose config`) |

Les 88 tests se répartissent ainsi :

| Fichier | Tests | Couverture |
|---------|-------|------------|
| `tests/test_api.py` | 28 | Prédictions (avec modèles entraînés à la volée), explications SHAP, batch, validation des entrées, webhook, règle d'alerte, anti-doublon, acquittement, notification, monitoring, fonctionnement sans base |
| `tests/test_regles.py` | 25 | Niveaux de risque aux seuils, règle d'alerte, rejet des configurations invalides, chargement d'un autre fichier de règles |
| `tests/test_model.py` | 19 | Features, rééquilibrage 50/50, paradoxe de l'accuracy, comptage des alertes par niveau, rejeu et évaluation de la règle d'alerte |
| `tests/test_data_preparation.py` | 16 | Features glissantes, encodage, nettoyage (doublons, hors bornes, capteur figé, imputation limitée, pas d'imputation entre machines) |

Les tests n'ont besoin ni des modèles entraînés ni d'un serveur PostgreSQL : ils tournent tels quels en CI.

### 7.5 Exploitation

- **Supervision** : `GET /health` (modèles chargés, base disponible), healthchecks Docker, journaux de l'API (chaque alerte y est tracée).
- **Ré-entraînement** : `python -m src.bootstrap --force` (ou `docker compose run --rm trainer python -m src.bootstrap --force`), puis redémarrage de l'API. Vérifier dans la page « Performance des modèles » que les métriques et la validation de la règle d'alerte restent au niveau attendu avant de remettre en service.
- **Ajustement des seuils** : modifier `config/regles_metier.yaml`, redémarrer l'API et le dashboard ; relancer l'entraînement pour recalculer la validation de la règle.

### 7.6 Perspectives d'industrialisation

| Évolution | Bénéfice |
|-----------|----------|
| **Features temporelles** calculées sur l'historique en base (tendances, moyennes glissantes) | Principal levier pour dater plus précisément la panne et réduire les alertes précoces |
| Gradient Boosting en production après validation sur données réelles | Gain de 2 à 3 points de F1 et un peu moins de fausses alertes |
| Ré-entraînement automatisé (pipeline MLOps) et suivi de la dérive des données | Maintien des performances sans intervention manuelle |
| Règles et modèles par site (variable `MECHA_REGLES`, données propres à chaque usine) | Adaptation aux équipements et aux pratiques de chaque site |
| Ingestion en flux (MQTT / Kafka) devant l'API | Montée en charge et découplage vis-à-vis des capteurs |
| Intégration GMAO : création automatique d'un ordre de travail à partir du webhook d'alerte | Boucle complète alerte → intervention → acquittement |
| Authentification, TLS, interface FR/ES | Prérequis du déploiement sur les sites espagnols et en production |

---

## 8. Enjeux RGPD et protection des données

### Contexte réglementaire

Bien que le prototype utilise des **données entièrement simulées** (aucune donnée personnelle), le déploiement en production chez MECHA impliquerait le traitement de données soumises au RGPD (Règlement Général sur la Protection des Données - UE 2016/679).

### Types de données concernées en production

| Type de donnée | Nature | Risque RGPD |
|----------------|--------|-------------|
| Données capteurs machines | Données industrielles (température, pression, vibration) | **Faible** — pas de données personnelles directes |
| Identifiants machines | Données techniques (machine_id, modèle, âge) | **Faible** — pas de lien direct avec une personne |
| Logs d'intervention maintenance | Peuvent contenir le nom du technicien, la date, le détail | **Moyen** — données personnelles indirectes |
| Logs d'accès au dashboard | Adresse IP, identifiant utilisateur, horodatage | **Moyen** — données personnelles |
| Planning des équipes | Noms, horaires, affectations | **Élevé** — données personnelles directes |

### Mesures recommandées pour le déploiement en production

1. **Anonymisation / pseudonymisation** : remplacer les noms de techniciens par des identifiants dans les logs, pseudonymiser les logs d'accès.
2. **Minimisation des données** : ne collecter que les données strictement nécessaires à la prédiction.
3. **Durée de conservation** : données capteurs brutes 2 ans, logs d'accès 6 mois.
4. **Sécurité technique** : chiffrement TLS, authentification obligatoire, solution on-premise.
5. **Rôle du DPO** : consultation avant déploiement, registre de traitement, AIPD si nécessaire.
6. **Droits des personnes** : information des techniciens, droit d'accès, rectification et effacement.

### Pour ce prototype

Le prototype utilise exclusivement un **dataset synthétique généré par script Python**. **Aucune donnée personnelle n'est collectée, traitée ou stockée**. Les mesures RGPD ci-dessus s'appliquent uniquement en cas de déploiement avec des données réelles. L'acquittement des alertes n'enregistre pas l'identité de la personne qui acquitte ; l'ajouter en production imposerait les mesures de pseudonymisation ci-dessus.
