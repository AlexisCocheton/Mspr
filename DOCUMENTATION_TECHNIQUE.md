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
│                    │  Collecte &     │  reference/                  │
│                    │  Simulation     │  generate_dataset.py         │
│                    │  (data/raw/)    │  → mecha_dataset_full.csv    │
│                    └────────┬────────┘                              │
│                             │                                       │
│                    ┌────────▼────────┐                              │
│                    │  Préparation    │  data_preparation.py         │
│                    │  des données    │  - Features glissantes 24h   │
│                    │  (ETL)          │  - Encodage catégoriels      │
│                    └────────┬────────┘  - Label panne_dans_24h      │
│                             │                                       │
│                    ┌────────▼────────┐                              │
│                    │  Entraînement   │  model_training.py           │
│                    │  des modèles    │  - 6 classifieurs x 2 tâches │
│                    │  ML             │  - RF + GB pour RUL          │
│                    └────────┬────────┘                              │
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

1. **Simulation** : `reference/generate_dataset.py` génère un dataset IoT synthétique (393 120 enregistrements horaires, 30 machines, 5 usines, 18 mois)
2. **Centralisation** : Les données brutes sont stockées dans `data/raw/mecha_dataset_full.csv`
3. **Préparation** : `data_preparation.py` ajoute les features glissantes 24h, encode les catégorielles, calcule le label `panne_dans_24h`
4. **Modélisation** : `model_training.py` entraîne 6 algorithmes sur 2 tâches de classification + 2 modèles de régression RUL
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
├── reference/
│   └── generate_dataset.py     # Générateur de données synthétiques MECHA
├── data/
│   ├── raw/                    # Dataset brut (mecha_dataset_full.csv)
│   └── processed/              # Dataset enrichi + category_maps.json
├── src/
│   ├── data_preparation.py     # ETL et feature engineering
│   ├── model_training.py       # Entraînement et évaluation ML
│   ├── api.py                  # API REST FastAPI
│   └── dashboard.py            # Dashboard Streamlit
├── models/                     # Modèles sauvegardés (.joblib)
├── tests/                      # Tests unitaires
├── notebooks/                  # Notebook d'exploration
├── docs/                       # Documentation métier
├── Dockerfile                  # Image Docker
├── docker-compose.yml          # Orchestration multi-services
├── .github/workflows/ci.yml    # Pipeline CI/CD
└── requirements.txt            # Dépendances Python
```

---

## 3. Dictionnaire de données

### 3.1 Dataset brut — `data/raw/mecha_dataset_full.csv`

**Source** : Dataset synthétique généré par `reference/generate_dataset.py`
**Volume** : 393 120 enregistrements (30 machines × 13 104 heures = 18 mois)
**Granularité** : Horaire

| Colonne | Type | Unité | Description |
|---------|------|-------|-------------|
| `timestamp` | datetime | — | Horodatage du relevé (toutes les heures) |
| `machine_id` | str | — | Identifiant unique de la machine (ex : MCH-FR01-001) |
| `usine_id` | str | — | Identifiant de l'usine (USN-FR-01 … USN-ES-02) |
| `type_machine` | str | — | Type de machine : CNC-Fraisage, CNC-Tournage, Découpe-Laser, Centre-Usinage |
| `temperature_C` | float | °C | Température de la machine mesurée par capteur thermique |
| `vibration_mm_s` | float | mm/s | Niveau de vibration mesuré par accéléromètre |
| `courant_A` | float | A | Courant électrique absorbé par le moteur |
| `pression_bar` | float | bar | Pression hydraulique dans le circuit |
| `vitesse_tr_min` | float | tr/min | Vitesse de rotation de la broche |
| `age_machine_h` | int | h | Âge total de la machine depuis sa mise en service |
| `h_depuis_maintenance` | int | h | Heures écoulées depuis la dernière intervention de maintenance |
| `score_degradation` | float | [0–1] | Score interne de dégradation (0 = neuf, 1 = défaillance imminente) |
| `rul_heures` | float | h | Remaining Useful Life : heures restantes avant la prochaine panne |
| `en_panne` | int | 0/1 | Indicateur de défaillance active à cet instant |
| `etat_machine` | str | — | État qualitatif : normal / a_risque / critique / en_panne |

### 3.2 Dataset préparé — `data/processed/mecha_unified_prepared.csv`

En plus des colonnes brutes, les colonnes suivantes sont ajoutées par `data_preparation.py` :

| Colonne | Type | Description |
|---------|------|-------------|
| `temperature_mean_24h` | float | Moyenne glissante 24h de la température par machine |
| `temperature_std_24h` | float | Écart-type glissant 24h de la température par machine |
| `vibration_mean_24h` | float | Moyenne glissante 24h des vibrations |
| `vibration_std_24h` | float | Écart-type glissant 24h des vibrations |
| `courant_mean_24h` | float | Moyenne glissante 24h du courant |
| `courant_std_24h` | float | Écart-type glissant 24h du courant |
| `pression_mean_24h` | float | Moyenne glissante 24h de la pression |
| `pression_std_24h` | float | Écart-type glissant 24h de la pression |
| `vitesse_mean_24h` | float | Moyenne glissante 24h de la vitesse |
| `vitesse_std_24h` | float | Écart-type glissant 24h de la vitesse |
| `type_machine_encoded` | int | Encodage label de type_machine (LabelEncoder) |
| `usine_encoded` | int | Encodage label de usine_id (LabelEncoder) |
| `panne_dans_24h` | int | 0/1 — cible prédictive : panne dans les 24h (rul_heures ≤ 24) |

### 3.3 Features utilisées pour l'entraînement ML (`FEATURE_COLS`)

Les features glissantes 24h sont calculées mais **exclues de l'entraînement** car elles créent un signal temporel parfait (la moyenne intègre déjà l'état actuel), rendant la classification triviale (F1 ≈ 1.0). En production, seule la mesure courante est disponible en temps réel.

| Feature | Description |
|---------|-------------|
| `temperature_C` | Capteur température instantané |
| `vibration_mm_s` | Capteur vibration instantané |
| `courant_A` | Capteur courant instantané |
| `pression_bar` | Capteur pression instantané |
| `vitesse_tr_min` | Capteur vitesse instantané |
| `age_machine_h` | Contexte machine — âge |
| `h_depuis_maintenance` | Contexte machine — dernière maintenance |
| `type_machine_encoded` | Catégorielle encodée |
| `usine_encoded` | Catégorielle encodée |

---

## 4. Modèles de Machine Learning

### 4.1 Tâche 1 — Classification : machine en panne (`en_panne`)

**Objectif** : Détecter si une machine est actuellement en défaillance.
**Déséquilibre** : 2.53% de positifs — compensé par `class_weight="balanced"`.

| Algorithme | F1 | Precision | Recall | AUC-ROC | CV F1 |
|------------|-----|-----------|--------|---------|-------|
| Random Forest | 0.9865 | 0.9960 | 0.9773 | 1.0000 | 0.9881 ± 0.003 |
| Gradient Boosting | 0.9906 | 0.9955 | 0.9857 | 0.9999 | 0.9879 ± 0.003 |
| Decision Tree | 0.9665 | 0.9499 | 0.9837 | 0.9918 | 0.9569 ± 0.003 |
| Logistic Regression | 0.9273 | 0.8678 | 0.9956 | 0.9999 | — |
| KNN | 0.9791 | 1.0000 | 0.9590 | 0.9933 | — |
| SVM | 0.9617 | 0.9503 | 0.9733 | 0.9996 | — |

### 4.2 Tâche 2 — Classification préventive : panne dans 24h (`panne_dans_24h`)

**Objectif** : Prédire si une panne surviendra dans les 24 prochaines heures.
**Déséquilibre** : 2.46% de positifs. Cette tâche est intrinsèquement plus difficile car elle prédit l'avenir sans historique temporel disponible en temps réel.

| Algorithme | F1 | Precision | Recall | AUC-ROC | CV F1 |
|------------|-----|-----------|--------|---------|-------|
| Random Forest | 0.8407 | 0.7294 | 0.9922 | 0.9993 | 0.8392 ± 0.004 |
| Gradient Boosting | 0.9635 | 0.9520 | 0.9752 | 0.9998 | 0.9605 ± 0.004 |
| Decision Tree | 0.7385 | 0.5886 | 0.9907 | 0.9893 | 0.7383 ± 0.006 |
| Logistic Regression | 0.6176 | 0.4490 | 0.9891 | 0.9961 | — |
| KNN | 0.9557 | 0.9165 | 0.9984 | 0.9996 | — |
| SVM | 0.7819 | 0.6423 | 0.9990 | 0.9988 | — |

### 4.3 Tâche 3 — Régression : temps restant avant panne (`rul_heures`)

**Objectif** : Estimer le Remaining Useful Life (RUL) en heures.

| Algorithme | MAE | RMSE | R² |
|------------|-----|------|----|
| Random Forest | 8.16 h | 12.23 h | 0.9974 |
| Gradient Boosting | 11.49 h | 15.97 h | 0.9956 |

### 4.4 Justification des choix algorithmiques

- **Random Forest** : modèle de référence, robuste au bruit et aux données déséquilibrées, interprétable via l'importance des features. Sélectionné pour l'API de production.
- **Gradient Boosting** : boosting séquentiel, généralement plus précis que RF mais plus lent à entraîner. Bon complément de comparaison.
- **Decision Tree** : modèle simple et interprétable, sert de baseline et permet une visualisation directe des règles de décision.
- **Logistic Regression** : baseline linéaire, permet de quantifier le gain apporté par les modèles non-linéaires.
- **KNN** : approche non-paramétrique, sensible aux données aberrantes. Inclus pour la diversité algorithmique.
- **SVM** : sous-échantillonnage à 10 000 observations pour des raisons de performance. Bon rappel sur les classes minoritaires.

### 4.5 Features les plus importantes (Random Forest)

Par ordre d'importance décroissante pour la tâche `en_panne` :
1. `temperature_C` — premier signal de dégradation thermique
2. `vitesse_tr_min` — chute de régime lors des défaillances
3. `pression_bar` — pression hydraulique anormale
4. `h_depuis_maintenance` — durée depuis la dernière intervention
5. `courant_A` — surcharge électrique en cas de blocage mécanique

---

## 5. API REST

### Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | État de l'API et liste des modèles chargés |
| GET | `/model/info` | Métriques d'entraînement et modèles disponibles |
| POST | `/predict/panne` | Prédit si la machine est actuellement en panne |
| POST | `/predict/panne24h` | Prédit si une panne surviendra dans les 24h |
| POST | `/predict/rul` | Estime le temps restant avant défaillance (RUL) |

### Format de requête (POST)

```json
{
  "temperature_C": 85.0,
  "vibration_mm_s": 3.1,
  "courant_A": 22.0,
  "pression_bar": 5.5,
  "vitesse_tr_min": 1350,
  "age_machine_h": 12000,
  "h_depuis_maintenance": 400,
  "type_machine": "CNC-Fraisage",
  "usine_id": "USN-FR-01"
}
```

### Règles métier (seuils d'alerte)

| Probabilité | Niveau de risque | Action recommandée |
|-------------|------------------|--------------------|
| < 30% | Faible | Aucune action |
| 30–50% | Moyen | Surveillance renforcée |
| 50–80% | Élevé | Maintenance préventive sous 24h |
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

Le pipeline CI/CD (`.github/workflows/ci.yml`) exécute automatiquement :
1. **Tests** : `pytest tests/ -v` sur Python 3.11 et 3.12
2. **Build Docker** : construction et vérification de l'image

### Exécution locale (sans Docker)

```bash
# 1. Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Générer le dataset synthétique
python reference/generate_dataset.py
cp mecha_dataset_full.csv data/raw/mecha_dataset_full.csv

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

---

## 7. Origine et hypothèses des données

### Source

Le dataset MECHA est **entièrement synthétique**, généré par `reference/generate_dataset.py`. Ce choix garantit la confidentialité (aucune donnée réelle d'usine) et permet de contrôler précisément les scénarios de défaillance.

| Paramètre | Valeur |
|-----------|--------|
| Période couverte | 01/01/2024 → 30/06/2025 (18 mois) |
| Granularité | Horaire |
| Machines | 30 (4 types : CNC-Fraisage, CNC-Tournage, Découpe-Laser, Centre-Usinage) |
| Usines | 5 (3 France, 2 Espagne) |
| Volume | 393 120 lignes × 17 colonnes |
| Taux de panne | ~2.5% |

### Hypothèses de simulation

Le générateur modélise les comportements physiques suivants :

- **Dégradation progressive** : chaque machine suit une courbe de dégradation (score 0→1) corrélée à l'usure et aux heures depuis la dernière maintenance.
- **Capteurs physiquement cohérents** :
  - Température et courant augmentent avec la dégradation (+45% entre état normal et état critique)
  - Pression et vitesse de rotation diminuent avec la dégradation (−30%)
  - Vibrations augmentent avec la dégradation
- **Variabilité climatique** : les sites espagnols subissent des variations de température ambiante plus importantes
- **Pannes planifiées** : chaque machine subit 2 à 5 pannes sur 18 mois, espacées de manière réaliste (minimum 168h entre deux pannes)
- **Maintenance** : des interventions rétablissent partiellement l'état des machines (reset du drift de maintenance)
- **Score capteurs pendant la panne** : le score de dégradation est forcé à 1.0 pendant la période de panne, reflétant que les capteurs indiquent une défaillance active maximale

### Limites connues

- Les corrélations entre capteurs sont linéaires (simplification)
- Pas de modélisation de pannes multi-causes simultanées
- Les données ne reflètent pas exactement les conditions d'une usine réelle spécifique
- Un historique minimal est nécessaire pour que les features glissantes 24h soient fiables (premières 24h par machine)

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

Le prototype utilise exclusivement un **dataset synthétique généré par script Python**. **Aucune donnée personnelle n'est collectée, traitée ou stockée**. Les mesures RGPD ci-dessus s'appliquent uniquement en cas de déploiement avec des données réelles.

---

## 9. Perspectives d'industrialisation

- **Streaming temps réel** : intégration avec Apache Kafka pour traiter les données capteurs en temps réel
- **Réentraînement automatique** : pipeline MLOps pour réentraîner périodiquement les modèles avec les nouvelles données
- **Adaptation par site** : modèles spécifiques à chaque usine via du transfer learning
- **Alerting** : intégration avec des systèmes de notification (email, SMS, SCADA)
- **Explicabilité** : intégration de SHAP pour expliquer chaque prédiction aux équipes maintenance
