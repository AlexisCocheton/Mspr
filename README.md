# MECHA — Maintenance Prédictive par Intelligence Artificielle

Prototype IA de maintenance prédictive pour l'entreprise industrielle MECHA (5 usines, ~100 machines critiques). Développé dans le cadre du MSPR TPRE841 — Bloc 4 — EPSI 2025-2026.

## Fonctionnalités

- **Classification** de l'état machine : normale / à risque (`en_panne`, `panne_dans_24h`)
- **Prédiction RUL** : estimation du temps restant avant défaillance (MAE = 8,16h, R² = 0.9974)
- **API REST FastAPI** avec documentation Swagger auto-générée
- **Dashboard Streamlit** avec 4 vues métier et niveaux d'alerte colorés
- **Détection d'anomalies** non supervisée (Isolation Forest)

## Démarrage rapide (Docker)

```bash
docker-compose up --build
```

- API : http://localhost:8000
- Dashboard : http://localhost:8501
- Documentation Swagger : http://localhost:8000/docs

## Démarrage local (sans Docker)

```bash
# 1. Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Générer le dataset synthétique
python reference/generate_dataset.py

# 4. Préparer les données
python src/data_preparation.py

# 5. Entraîner les modèles (~10 min)
python src/model_training.py

# 6. Lancer l'API
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 7. Lancer le dashboard (autre terminal)
streamlit run src/dashboard.py

# 8. Lancer les tests
pytest tests/ -v
```

## Structure du projet

```
mspr/
├── reference/                  # Générateur de données synthétiques
├── data/
│   ├── raw/                    # Dataset brut (mecha_dataset_full.csv)
│   └── processed/              # Dataset enrichi + encodages
├── src/
│   ├── data_preparation.py     # ETL et feature engineering
│   ├── model_training.py       # Entraînement et évaluation ML
│   ├── api.py                  # API REST FastAPI
│   └── dashboard.py            # Dashboard Streamlit
├── models/                     # Modèles entraînés (.joblib)
├── tests/                      # 23 tests unitaires pytest
├── notebooks/                  # Notebook d'exploration EDA
├── docs/                       # Documentation complète
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml    # CI/CD GitHub Actions
```

## Documentation

| Document | Description |
|----------|-------------|
| [Documentation technique](DOCUMENTATION_TECHNIQUE.md) | Architecture, modèles, API, déploiement, RGPD |
| [Rapport de validation et déploiement](docs/RAPPORT_VALIDATION_DEPLOIEMENT.md) | Métriques, tests, projection industrielle |
| [Guide utilisateur métier](docs/GUIDE_UTILISATEUR_METIER.md) | Comment utiliser le dashboard et interpréter les alertes |
| [Conduite du changement](docs/CONDUITE_DU_CHANGEMENT.md) | Plan de formation et déploiement progressif |
| [Compte-rendu entretien client](docs/COMPTE_RENDU_ENTRETIEN_CLIENT.md) | Besoins fonctionnels collectés |
| [Support de soutenance](docs/SUPPORT_SOUTENANCE.md) | Slides de présentation |

## Stack technique

Python 3.12 · scikit-learn · FastAPI · Streamlit · Docker · GitHub Actions · pytest
