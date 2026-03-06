# Support de Soutenance — MECHA Maintenance Prédictive IA

*MSPR TPRE841 — Bloc 4 — EPSI 2025-2026*

---

## Slide 1 : Page de titre

**MECHA — Solution de Maintenance Prédictive par Intelligence Artificielle**

Conception et réalisation d'un prototype IA pour la prédiction de pannes machines dans un contexte industriel multi-sites.

Équipe projet : [Noms des 4 membres]
Date : [Date de soutenance]

---

## Slide 2 : Contexte et problématique

**L'entreprise MECHA** :
- Fabricant de pièces mécaniques haute précision (aéronautique, automobile)
- 5 usines (3 France, 2 Espagne), ~100 machines critiques
- Capteurs IoT déployés (température, pression, vitesse, vibration)

**Problématique** :
- Les arrêts non planifiés coûtent ~15% de capacité de production
- Maintenance actuelle : corrective + préventive planifiée (insuffisant)
- Besoin : **anticiper les pannes 24h avant** pour planifier les interventions

---

## Slide 3 : Objectifs du projet

1. **Classifier** l'état des machines : fonctionnement normal vs. à risque
2. **Prédire** le temps restant avant défaillance (RUL)
3. **Fournir** un outil opérationnel aux équipes maintenance (dashboard + API)
4. **Conteneuriser** la solution pour un déploiement multi-sites
5. **Documenter** et préparer l'intégration dans le SI existant

---

## Slide 4 : Architecture technique

```
Capteurs IoT → Collecte (data/raw/) → Préparation (ETL) → Modélisation (ML)
                                                               ↓
                                          ┌──────────────────────────────────┐
                                          │   API FastAPI    │  Dashboard   │
                                          │   (port 8000)    │  Streamlit   │
                                          │   5 endpoints    │  (port 8501) │
                                          └──────────────────────────────────┘
                                          │         Docker / docker-compose          │
                                          │         GitHub Actions (CI/CD)           │
```

**Stack technique** : Python 3.12 | scikit-learn | FastAPI | Streamlit | Docker | GitHub Actions

---

## Slide 5 : Les données

### Deux datasets publics adaptés au contexte MECHA

| Dataset | Source | Volume | Contenu |
|---------|--------|--------|---------|
| AI4I 2020 | Kaggle (CC BY 4.0) | 10 000 lignes | Capteurs machines + types de pannes |
| Azure PdM | Microsoft (MIT) | 876 100 lignes | Télémétrie horaire (100 machines/1 an) + erreurs + pannes + maintenance |

### Travail de préparation réalisé

- Renommage des colonnes → vocabulaire MECHA
- Conversion des unités (Kelvin → Celsius)
- **Feature engineering** :
  - Moyennes et écarts-types glissants 24h
  - Puissance mécanique, ratio couple/vitesse
  - Heures depuis dernière maintenance par composant
  - Comptage d'erreurs par type
- **Labellisation** : `failure_within_24h` (classif.) + `hours_to_failure` (RUL)

---

## Slide 6 : Modèles de Machine Learning

### Algorithmes utilisés

| Modèle | Rôle | Justification |
|--------|------|---------------|
| **Random Forest Classifier** | Classif. normal/à risque | Robuste au bruit, gère le déséquilibre, interprétable |
| **Random Forest Regressor** | Prédiction RUL | Bon compromis performance/interprétabilité |
| **Logistic Regression** | Baseline comparaison | Valider que le RF apporte un vrai gain |

### Gestion du déséquilibre des classes

- Seulement ~3% de pannes dans les données
- Solution : paramètre `class_weight="balanced"` pour sur-pondérer la classe minoritaire

---

## Slide 7 : Résultats — Classification

| Métrique | AI4I (RF) | PdM (RF) | AI4I (LR baseline) | PdM (LR baseline) |
|----------|-----------|----------|---------------------|--------------------|
| Accuracy | 99.05% | 99.36% | 86.25% | 89.85% |
| Precision | **90.16%** | 75.90% | 18.35% | 15.54% |
| Recall | 80.88% | **98.89%** | 88.24% | 94.15% |
| F1-Score | **85.27%** | **85.89%** | 30.38% | 26.68% |
| AUC-ROC | 96.43% | **99.94%** | 93.34% | 96.06% |

**Conclusion** : Le Random Forest surpasse massivement la baseline (~85% vs ~30% de F1).
Le recall de 98.9% sur PdM signifie que **quasiment toutes les pannes sont détectées**.

---

## Slide 8 : Résultats — Prédiction RUL

| Métrique | Valeur | Interprétation |
|----------|--------|---------------|
| MAE | **0.56 heures** | Erreur moyenne de ~34 minutes |
| RMSE | 1.43 heures | Erreurs maximales contenues |
| R² | **0.9577** | 96% de la variance expliquée |

Le modèle estime le temps restant avant panne avec une précision de ±34 minutes, ce qui est **largement suffisant** pour planifier une intervention de maintenance.

---

## Slide 9 : Solution applicative — Dashboard

**4 vues métier** :

1. **Vue d'ensemble** : KPI globaux, distribution des alertes, évolution temporelle
2. **Prédiction temps réel** : saisie des capteurs → prédiction instantanée avec jauge de risque
3. **Analyse des données** : historique capteurs par machine, matrice de corrélation
4. **Performance modèles** : comparaison RF vs LR, métriques, visualisations

**Niveaux d'alerte** :
- Vert (< 30%) → Normal
- Jaune (30-50%) → Surveillance
- Orange (50-80%) → Maintenance préventive
- Rouge (> 80%) → Arrêt recommandé

---

## Slide 10 : API REST

5 endpoints FastAPI avec documentation Swagger auto-générée :

```
GET  /health          → État de l'API
GET  /model/info      → Métriques des modèles
POST /predict/ai4i    → Prédiction panne (AI4I)
POST /predict/pdm     → Prédiction panne 24h (PdM)
POST /predict/rul     → Estimation temps restant
```

Chaque réponse inclut : **prédiction + probabilité + niveau de risque + recommandation**

---

## Slide 11 : Déploiement et CI/CD

### Conteneurisation
- **Dockerfile** : image Python 3.12 avec toutes les dépendances
- **docker-compose** : 2 services (API + Dashboard) lancés en une commande

### Intégration continue (GitHub Actions)
- Tests automatiques sur Python 3.11 et 3.12
- Build Docker vérifié à chaque push

### Stratégie de déploiement multi-sites
1. **Mois 1-2** : Pilote sur 1 site (mode observation)
2. **Mois 3-4** : Extension aux 3 sites France (mode opérationnel)
3. **Mois 5-6** : Déploiement sur les 2 sites Espagne

---

## Slide 12 : Validation de la solution

### Validation technique
- **16 tests unitaires** (pytest) : tous passent
- Validation croisée 5-fold : F1 stable ~85% (faible variance)
- Comparaison RF vs baseline : gain de +55 points de F1

### Validation fonctionnelle
- Dashboard testé avec données réalistes
- API testée avec des requêtes valides et invalides (erreurs 422)
- Seuils d'alerte calibrés sur les distributions des probabilités

### Limites identifiées
- Données simulées (pas de conditions réelles)
- Déséquilibre des classes à surveiller en production
- Nécessite un historique 24h pour les moyennes glissantes

---

## Slide 13 : Conduite du changement

| Phase | Action | Durée |
|-------|--------|-------|
| Formation référents | 1 référent/site formé sur l'administration | 1 jour |
| Formation équipes | Sessions de 2h par groupe de 10 techniciens | 2 semaines |
| Accompagnement terrain | Référent présent sur site | 2 semaines |
| Retour d'expérience | Sessions hebdomadaires 30 min | 1 mois |

**Cible** : 85% d'adoption à 6 mois, -20% d'arrêts non planifiés

---

## Slide 14 : RGPD et sécurité

- Prototype actuel : **aucune donnée personnelle** (datasets publics)
- En production : anonymisation des logs maintenance, pseudonymisation des accès
- Données on-premise, chiffrement TLS, authentification obligatoire
- Consultation du DPO requise avant déploiement avec données réelles

---

## Slide 15 : Conclusion et perspectives

### Ce qui a été livré
- Pipeline complet : données → modèles → API → dashboard → Docker → CI/CD
- Modèles performants (F1 85%, Recall 99%, RUL MAE 34 min)
- Documentation complète (technique + utilisateur + conduite du changement)

### Perspectives d'évolution
- Streaming temps réel (Apache Kafka)
- Réentraînement automatique (MLOps)
- Adaptation multi-sites (transfer learning)
- Intégration alerting (email, SMS, SCADA)

**Merci — Questions ?**
