# MECHA — Maintenance prédictive par intelligence artificielle

Prototype de maintenance prédictive pour MECHA, fabricant de pièces mécaniques de précision (5 usines : 3 en France, 2 en Espagne). Projet MSPR TPRE841 — Bloc 4 — EPSI 2025-2026.

La solution reçoit les relevés horaires des capteurs machines, estime le risque de panne dans les 24 h, déclenche des alertes selon des règles métier paramétrables et restitue le tout dans un dashboard destiné aux équipes de maintenance et de production.

## Fonctionnalités

- **Classification de l'état machine** : probabilité de panne dans les 24 h et niveau de risque (faible / moyen / élevé / critique).
- **Temps restant avant défaillance (RUL)** : ordre de grandeur pour prioriser les interventions.
- **Détection d'anomalies** non supervisée (Isolation Forest).
- **Explication de chaque prédiction** : les 3 mesures qui ont le plus pesé (SHAP).
- **Alertes** : déclenchées après plusieurs relevés consécutifs à risque, acquittables, notifiables vers un outil externe (webhook).
- **Paramétrage métier sans code** : seuils, règles d'alerte, bornes capteurs et fréquences dans `config/regles_metier.yaml`.
- **Qualité des données** : contrôle de plausibilité à l'ingestion, nettoyage du fichier brut (doublons, valeurs aberrantes, capteurs figés, trous de mesure).
- **API REST FastAPI** (15 endpoints, Swagger), **PostgreSQL**, **dashboard Streamlit** (5 pages), **simulateur IoT**, **Docker** et **CI GitHub Actions**.

## Démarrage rapide (Docker)

```bash
cp .env.example .env     # optionnel : identifiants PostgreSQL, URL de notification des alertes
docker compose up --build
```

Au **premier lancement**, le service `trainer` prépare les données et entraîne les modèles (10 à 15 minutes) avant le démarrage de l'API ; les lancements suivants réutilisent les modèles de `./models`.

- Dashboard : http://localhost:8501
- API : http://localhost:8000 — documentation Swagger : http://localhost:8000/docs

Le simulateur envoie aussitôt les relevés des 30 machines ; les alertes apparaissent dans la page « Monitoring temps réel ».

## Démarrage local (sans Docker)

```bash
python -m venv venv
venv\Scripts\activate                 # Windows  (source venv/bin/activate sous Linux/Mac)
pip install -r requirements.txt

python reference/generate_dataset.py  # facultatif : régénère data/raw/mecha_dataset_full.csv (versionné)
python src/data_preparation.py        # nettoyage + features
python src/model_training.py          # entraînement et évaluations (10 à 15 min)

uvicorn src.api:app --port 8000       # API
streamlit run src/dashboard.py        # dashboard (autre terminal)
pytest tests/ -v                      # 88 tests
```

Le monitoring temps réel nécessite PostgreSQL (`docker compose up -d postgres`) et le simulateur (`python src/sensor_simulator.py` avec `API_URL=http://localhost:8000`).

## Résultats clés

| Indicateur | Valeur | Lecture |
|------------|--------|---------|
| Pannes signalées dans les 24 h qui précèdent | **75 %** | Règle configurée : risque « critique » sur 3 relevés consécutifs |
| Alertes suivies d'une panne dans les 7 jours | **90 %** | Seulement 26 % dans les 24 h : l'alerte est fiable sur la machine, moins sur le moment de la panne |
| Volume d'alertes | **0,53 par machine et par semaine** | Charge supportable pour les équipes |
| Erreur moyenne du RUL | **≈ 105 h** | Évaluée sur la période la plus récente : un ordre de grandeur, pas une échéance |

Ces chiffres viennent d'une validation par rejeu temporel : modèle entraîné sur le passé, rejoué heure par heure sur les 3 derniers mois. Le détail, y compris l'évaluation sur test 50/50 et à la distribution réelle, est dans la [documentation technique](DOCUMENTATION_TECHNIQUE.md) et le [rapport de validation](docs/RAPPORT_VALIDATION_DEPLOIEMENT.md).

## Structure du projet

```
mspr/
├── config/regles_metier.yaml   # Paramétrage métier : seuils, alertes, bornes, fréquences
├── reference/generate_dataset.py  # Générateur du dataset synthétique (brut, avec défauts)
├── data/
│   ├── raw/                    # mecha_dataset_full.csv (brut)
│   └── processed/              # Dataset nettoyé, encodages, bilan qualité
├── src/
│   ├── regles.py               # Lecture et validation des règles métier
│   ├── data_preparation.py     # Nettoyage et feature engineering
│   ├── model_training.py       # Entraînement, évaluations, validation de la règle d'alerte
│   ├── bootstrap.py            # Entraîne si les modèles sont absents (service trainer)
│   ├── api.py                  # API REST FastAPI
│   ├── database.py             # Accès PostgreSQL
│   ├── dashboard.py            # Dashboard Streamlit
│   └── sensor_simulator.py     # Simulateur IoT
├── db/init.sql                 # Schéma PostgreSQL (lectures, prédictions, alertes)
├── models/                     # Modèles entraînés (générés, non versionnés)
├── notebooks/exploration_mecha.ipynb  # Analyse exploratoire du dataset
├── tests/                      # test_api, test_regles, test_model, test_data_preparation
├── docs/                       # Documentation métier et validation
├── Dockerfile, docker-compose.yml, .env.example
└── .github/workflows/ci.yml    # Tests + build de l'image Docker
```

## Documentation

| Document | Contenu |
|----------|---------|
| [Documentation technique](DOCUMENTATION_TECHNIQUE.md) | Architecture, données, modèles, solution applicative, déploiement, RGPD |
| [Rapport de validation et déploiement](docs/RAPPORT_VALIDATION_DEPLOIEMENT.md) | Démarche de validation, résultats, projection de déploiement dans les usines |
| [Guide utilisateur métier](docs/GUIDE_UTILISATEUR_METIER.md) | Utiliser le dashboard, interpréter les alertes et leurs limites |
| [Conduite du changement](docs/CONDUITE_DU_CHANGEMENT.md) | Acteurs, messages, formation, déploiement progressif |
| [Compte-rendu entretien client](docs/COMPTE_RENDU_ENTRETIEN_CLIENT.md) | Guide d'entretien et synthèse des besoins |
| [Prompts IA générative](docs/PROMPTS_IA.md) | Prompts utilisés et intégration des résultats |
| [Notebook d'exploration](notebooks/exploration_mecha.ipynb) | Analyse exploratoire du dataset MECHA |

## Stack technique

Python 3.11 / 3.12 · pandas · scikit-learn · SHAP · FastAPI · PostgreSQL · SQLAlchemy · Streamlit · Plotly · Docker · GitHub Actions · pytest
