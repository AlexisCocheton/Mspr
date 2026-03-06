# Documentation Technique - MECHA Maintenance Prédictive

## 1. Schéma d'architecture de la solution

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE GLOBALE MECHA                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Capteurs IoT │    │  Historiques  │    │  Systèmes SCADA/MES │  │
│  │  (machines)   │    │  maintenance │    │  (supervision)       │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
│         │                   │                        │              │
│         └───────────────────┼────────────────────────┘              │
│                             │                                       │
│                    ┌────────▼────────┐                              │
│                    │  Collecte &     │                              │
│                    │  Centralisation │                              │
│                    │  (data/raw/)    │                              │
│                    └────────┬────────┘                              │
│                             │                                       │
│                    ┌────────▼────────┐                              │
│                    │  Préparation    │  data_preparation.py         │
│                    │  des données    │  - Nettoyage                 │
│                    │  (ETL)          │  - Feature engineering       │
│                    └────────┬────────┘  - Labellisation             │
│                             │                                       │
│                    ┌────────▼────────┐                              │
│                    │  Entraînement   │  model_training.py           │
│                    │  des modèles    │  - Random Forest (classif.)  │
│                    │  ML             │  - Random Forest (régression)│
│                    └────────┬────────┘  - Logistic Regression       │
│                             │                                       │
│              ┌──────────────┼──────────────┐                       │
│              │              │              │                        │
│     ┌────────▼──────┐ ┌────▼─────┐ ┌─────▼──────────┐            │
│     │  API FastAPI   │ │ Modèles  │ │  Dashboard     │            │
│     │  (prédictions) │ │ (.joblib)│ │  Streamlit     │            │
│     │  Port 8000     │ │          │ │  Port 8501     │            │
│     └───────────────┘ └──────────┘ └────────────────┘             │
│                                                                     │
│     ┌──────────────────────────────────────────────────┐           │
│     │  Docker / docker-compose (conteneurisation)       │           │
│     │  GitHub Actions (CI/CD)                           │           │
│     └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

### Flux de données

1. **Collecte** : Les données capteurs (température, vitesse, pression, vibration) sont collectées toutes les heures depuis les machines de production
2. **Centralisation** : Les données brutes sont stockées dans `data/raw/`
3. **Préparation** : Le module `data_preparation.py` nettoie, fusionne et enrichit les données
4. **Modélisation** : Le module `model_training.py` entraîne les modèles de ML
5. **Exposition** : L'API REST (`api.py`) expose les prédictions via des endpoints HTTP
6. **Visualisation** : Le dashboard Streamlit (`dashboard.py`) affiche les résultats aux équipes métiers

---

## 2. Choix techniques

### Langages et outils

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| Langage | Python 3.12+ | Écosystème ML/Data Science mature, large communauté |
| ML | scikit-learn | Bibliothèque robuste pour Random Forest, métriques, pipelines |
| Data | pandas, numpy | Standard pour la manipulation de données tabulaires |
| API | FastAPI | Performances élevées, documentation auto-générée (Swagger) |
| Dashboard | Streamlit + Plotly | Prototypage rapide d'interfaces interactives |
| Conteneurisation | Docker + docker-compose | Déploiement standardisé multi-services |
| CI/CD | GitHub Actions | Intégration native avec Git, tests automatisés |
| Tests | pytest | Framework de test standard Python |

### Structure du code

```
mspr/
├── data/
│   ├── raw/                    # Données brutes (CSV originaux)
│   └── processed/              # Données nettoyées et enrichies
├── src/
│   ├── data_preparation.py     # ETL et feature engineering
│   ├── model_training.py       # Entraînement et évaluation ML
│   ├── api.py                  # API REST FastAPI
│   └── dashboard.py            # Dashboard Streamlit
├── models/                     # Modèles sauvegardés (.joblib)
├── tests/                      # Tests unitaires
├── notebooks/                  # Notebook d'exploration
├── Dockerfile                  # Image Docker
├── docker-compose.yml          # Orchestration multi-services
├── .github/workflows/ci.yml    # Pipeline CI/CD
└── requirements.txt            # Dépendances Python
```

---

## 3. Dictionnaire de données

### 3.1 Dataset AI4I 2020 (après préparation)

**Source** : AI4I 2020 Predictive Maintenance Dataset (Kaggle)
**Fichier** : `mecha_ai4i_prepared.csv`
**Volume** : 10 000 enregistrements

| Colonne | Type | Unité | Description |
|---------|------|-------|-------------|
| record_id | int | - | Identifiant unique de l'enregistrement |
| product_id | str | - | Identifiant produit (ex: M14860) |
| quality_type | str | L/M/H | Type de qualité du produit (Low/Medium/High) |
| air_temp_k | float | K | Température ambiante en Kelvin |
| process_temp_k | float | K | Température du process en Kelvin |
| rotational_speed_rpm | float | RPM | Vitesse de rotation de l'outil |
| torque_nm | float | Nm | Couple exercé sur la pièce |
| tool_wear_min | int | min | Usure cumulée de l'outil |
| machine_failure | int | 0/1 | Indicateur de panne (cible principale) |
| tool_wear_failure | int | 0/1 | Panne par usure outil |
| heat_dissipation_failure | int | 0/1 | Panne par dissipation thermique |
| power_failure | int | 0/1 | Panne par surpuissance |
| overstrain_failure | int | 0/1 | Panne par surcharge |
| random_failure | int | 0/1 | Panne aléatoire |
| **air_temp_c** | float | °C | Température ambiante (dérivée) |
| **process_temp_c** | float | °C | Température process (dérivée) |
| **temp_diff** | float | °C | Écart thermique process - air (dérivée) |
| **power_w** | float | W | Puissance mécanique = couple × vitesse ang. (dérivée) |
| **torque_speed_ratio** | float | - | Ratio couple/vitesse (indicateur de surcharge, dérivée) |

### 3.2 Dataset Azure PdM (après préparation)

**Source** : Microsoft Azure Predictive Maintenance Dataset
**Fichier** : `mecha_pdm_prepared.csv`
**Volume** : 876 100 enregistrements (100 machines × 8 761 heures)

| Colonne | Type | Unité | Description |
|---------|------|-------|-------------|
| datetime | datetime | - | Horodatage du relevé (horaire) |
| machine_id | int | - | Identifiant de la machine (1-100) |
| voltage | float | V | Tension mesurée |
| rotation_speed | float | RPM | Vitesse de rotation |
| pressure | float | bar | Pression mesurée |
| vibration | float | mm/s | Niveau de vibration |
| volt_mean_24h | float | V | Moyenne glissante 24h de la tension |
| volt_std_24h | float | V | Écart-type glissant 24h de la tension |
| rotate_mean_24h | float | RPM | Moyenne glissante 24h de la rotation |
| rotate_std_24h | float | RPM | Écart-type glissant 24h de la rotation |
| pressure_mean_24h | float | bar | Moyenne glissante 24h de la pression |
| pressure_std_24h | float | bar | Écart-type glissant 24h de la pression |
| vibration_mean_24h | float | mm/s | Moyenne glissante 24h des vibrations |
| vibration_std_24h | float | mm/s | Écart-type glissant 24h des vibrations |
| errorID_error1..5 | int | 0/1 | Erreur de type 1 à 5 survenue à cet instant |
| hours_since_maint_comp1..4 | int | h | Heures depuis dernière maintenance composant 1-4 |
| machine_model | str | - | Modèle de la machine (model1-4) |
| machine_age_years | int | ans | Âge de la machine |
| **failure_within_24h** | int | 0/1 | Panne dans les 24h suivantes (cible classif.) |
| **failure_component** | str | - | Composant qui va tomber en panne |
| **hours_to_failure** | float | h | Heures restantes avant panne (cible régression/RUL) |

---

## 4. Modèles de Machine Learning

### 4.1 Classification : état de la machine (normal / à risque)

**Objectif** : Prédire si une machine va tomber en panne dans les 24 prochaines heures.

#### Random Forest Classifier

| Métrique | AI4I | PdM |
|----------|------|-----|
| Accuracy | 99.05% | 99.36% |
| Precision | 90.16% | 75.90% |
| Recall | 80.88% | 98.89% |
| F1-Score | 85.27% | 85.89% |
| AUC-ROC | 96.43% | 99.94% |
| CV F1 (5-fold) | 80.82% | 85.90% |

#### Logistic Regression (baseline)

| Métrique | AI4I | PdM |
|----------|------|-----|
| Accuracy | 86.25% | 89.85% |
| F1-Score | 30.38% | 26.68% |
| AUC-ROC | 93.34% | 96.06% |

**Analyse** : Le Random Forest surpasse largement la régression logistique, particulièrement en termes de F1-Score. Le recall élevé sur PdM (98.89%) est crucial en contexte industriel car il minimise les pannes non détectées.

#### Justification du choix algorithmique

- **Random Forest** : choisi pour sa capacité à gérer les données déséquilibrées (3-4% de pannes), sa robustesse au bruit, et l'interprétabilité via l'importance des features
- **Logistic Regression** : utilisée comme baseline pour valider que le Random Forest apporte un gain significatif
- **class_weight="balanced"** : compense le déséquilibre des classes (97% normal vs 3% pannes)

### 4.2 Régression : temps restant avant défaillance (RUL)

**Objectif** : Estimer le nombre d'heures avant la prochaine panne.

| Métrique | Valeur |
|----------|--------|
| MAE | 0.56 heures |
| RMSE | 1.43 heures |
| R² | 0.9577 |

**Analyse** : Le modèle prédit le RUL avec une erreur moyenne de ~34 minutes, ce qui est excellent pour la planification de maintenance.

### 4.3 Features les plus importantes

Les features les plus discriminantes pour la détection de pannes sont (par ordre d'importance) :
- Usure outil / heures depuis dernière maintenance
- Couple (torque)
- Vitesse de rotation
- Puissance mécanique (feature dérivée)
- Écarts-types des capteurs sur 24h (variabilité = signe de dégradation)

---

## 5. API REST

### Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | État de l'API et des modèles |
| GET | `/model/info` | Informations et métriques des modèles |
| POST | `/predict/ai4i` | Prédiction de panne (dataset AI4I) |
| POST | `/predict/pdm` | Prédiction de panne dans 24h (dataset PdM) |
| POST | `/predict/rul` | Prédiction du temps restant avant panne |

### Règles métier (seuils d'alerte)

| Probabilité | Niveau de risque | Action recommandée |
|-------------|------------------|--------------------|
| < 30% | Faible | Aucune action |
| 30-50% | Moyen | Surveillance renforcée |
| 50-80% | Élevé | Maintenance préventive sous 24h |
| > 80% | Critique | Arrêt immédiat recommandé |

---

## 6. Déploiement

### Conteneurisation Docker

```bash
# Construire et lancer tous les services
docker-compose up --build

# API accessible sur http://localhost:8000
# Dashboard accessible sur http://localhost:8501
# Documentation API (Swagger) sur http://localhost:8000/docs
```

### CI/CD (GitHub Actions)

Le pipeline CI/CD exécute automatiquement :
1. **Tests** : `pytest tests/ -v` sur Python 3.11 et 3.12
2. **Build Docker** : construction et vérification de l'image

### Exécution locale (sans Docker)

```bash
# 1. Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Préparer les données
python src/data_preparation.py

# 4. Entraîner les modèles
python src/model_training.py

# 5. Lancer l'API
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 6. Lancer le dashboard (dans un autre terminal)
streamlit run src/dashboard.py

# 7. Lancer les tests
pytest tests/ -v
```

---

## 7. Origine et hypothèses des données

### Sources

| Dataset | Source | Licence |
|---------|--------|---------|
| AI4I 2020 | Kaggle - AI4I 2020 Predictive Maintenance | CC BY 4.0 |
| Azure PdM | Microsoft Azure - Predictive Maintenance | MIT |

### Hypothèses de simulation / adaptation

- Les colonnes ont été renommées pour correspondre au vocabulaire MECHA
- Les températures Kelvin ont été converties en Celsius
- Des features dérivées ont été créées (puissance, ratios, moyennes glissantes)
- Le label `failure_within_24h` a été construit en marquant les 24 heures précédant chaque panne
- Le RUL (Remaining Useful Life) a été calculé comme le temps en heures avant la prochaine défaillance connue

### Limites connues

- Les données sont simulées et ne reflètent pas exactement les conditions réelles des usines MECHA
- Le dataset AI4I n'a pas de dimension temporelle (pas de séries temporelles)
- Le déséquilibre des classes (~3% de pannes) peut influencer les prédictions en production
- Les moyennes glissantes 24h nécessitent un historique suffisant pour être fiables

---

## 8. Enjeux RGPD et protection des données

### Contexte réglementaire

Bien que le prototype utilise des **données simulées/publiques** (aucune donnée personnelle), le déploiement en production chez MECHA impliquerait le traitement de données soumises au RGPD (Règlement Général sur la Protection des Données - UE 2016/679).

### Types de données concernées en production

| Type de donnée | Nature | Risque RGPD |
|----------------|--------|-------------|
| Données capteurs machines | Données industrielles (température, pression, vibration) | **Faible** — pas de données personnelles directes |
| Identifiants machines | Données techniques (machine_id, modèle, âge) | **Faible** — pas de lien direct avec une personne |
| Logs d'intervention maintenance | Peuvent contenir le nom du technicien, la date, le détail | **Moyen** — données personnelles indirectes |
| Logs d'accès au dashboard | Adresse IP, identifiant utilisateur, horodatage | **Moyen** — données personnelles |
| Planning des équipes | Noms, horaires, affectations | **Élevé** — données personnelles directes |

### Mesures recommandées pour le déploiement en production

1. **Anonymisation / pseudonymisation** :
   - Remplacer les noms de techniciens par des identifiants anonymes dans les logs de maintenance
   - Ne pas stocker de données nominatives dans les datasets d'entraînement des modèles
   - Pseudonymiser les logs d'accès au dashboard

2. **Minimisation des données** :
   - Ne collecter que les données strictement nécessaires à la prédiction (capteurs + historique machines)
   - Ne pas croiser les données de production avec les données RH

3. **Durée de conservation** :
   - Données capteurs brutes : 2 ans (nécessaire pour le réentraînement des modèles)
   - Logs d'accès : 6 mois
   - Modèles entraînés : conservés tant qu'ils sont en production

4. **Sécurité technique** :
   - Chiffrement des données au repos et en transit (TLS)
   - Authentification obligatoire pour accéder à l'API et au dashboard
   - Cloisonnement réseau : la solution reste on-premise, pas d'envoi vers le cloud

5. **Rôle du DPO (Délégué à la Protection des Données)** :
   - Le DPO de MECHA doit être consulté avant le déploiement en production
   - Un registre de traitement doit être tenu à jour
   - Une analyse d'impact (AIPD) est recommandée si les logs de maintenance contiennent des données personnelles

6. **Droits des personnes** :
   - Les techniciens dont les données apparaissent dans les logs doivent être informés
   - Droit d'accès, de rectification et d'effacement applicable sur les données personnelles
   - Le traitement doit être fondé sur l'intérêt légitime de l'employeur (optimisation de la production)

### Pour ce prototype

Le prototype actuel utilise exclusivement des **datasets publics simulés** (AI4I 2020 sous licence CC BY 4.0, Azure PdM sous licence MIT). **Aucune donnée personnelle n'est collectée, traitée ou stockée**. Les mesures RGPD ci-dessus s'appliquent uniquement en cas de déploiement avec des données réelles.

---

## 9. Perspectives d'industrialisation

- **Streaming temps réel** : intégration avec Apache Kafka pour traiter les données capteurs en temps réel
- **Réentraînement automatique** : pipeline MLOps pour réentraîner périodiquement les modèles avec les nouvelles données
- **Multi-sites** : adaptation du modèle pour chaque usine avec du transfer learning
- **Alerting** : intégration avec des systèmes de notification (email, SMS, SCADA)
