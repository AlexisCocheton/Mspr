# Rapport de Validation et Projection de Déploiement Industriel
## MECHA — Solution de Maintenance Prédictive par IA

*MSPR TPRE841 — Bloc 4 — EPSI 2025-2026*

---

## 1. Objectifs de la validation

La validation de la solution MECHA a poursuivi deux axes complémentaires :

1. **Validation technique** : vérifier que les modèles ML atteignent des performances suffisantes pour un usage industriel et que les composants logiciels (API, dashboard, pipeline de données) fonctionnent de manière fiable.
2. **Validation fonctionnelle** : vérifier que la solution répond aux besoins exprimés par les équipes MECHA lors de l'entretien client (anticipation des pannes 24h avant, interface opérationnelle, niveaux d'alerte actionnables).

---

## 2. Démarche de validation technique

### 2.1 Protocole d'évaluation des modèles

Les modèles ont été évalués selon le protocole suivant :

- **Séparation train/test** : 80% entraînement / 20% test, avec stratification pour préserver les proportions de classes.
- **Validation croisée 5-fold** : appliquée aux deux meilleurs modèles (Random Forest et Gradient Boosting) pour mesurer la stabilité des performances.
- **Métriques retenues** : F1-Score (métrique principale compte tenu du déséquilibre des classes), Precision, Recall, AUC-ROC, MAE et R² pour la régression RUL.
- **Baseline de comparaison** : Régression Logistique utilisée comme référence linéaire pour quantifier l'apport des modèles non-linéaires.

### 2.2 Résultats — Tâche 1 : détection de panne active (`en_panne`)

**Contexte** : 2,53% de positifs (déséquilibre compensé par `class_weight="balanced"`).

| Algorithme | F1 | Precision | Recall | AUC-ROC | CV F1 (5-fold) |
|------------|-----|-----------|--------|---------|----------------|
| **Random Forest** | **0.9865** | 0.9960 | 0.9773 | 1.0000 | 0.9881 ± 0.003 |
| Gradient Boosting | 0.9906 | 0.9955 | 0.9857 | 0.9999 | 0.9879 ± 0.003 |
| Decision Tree | 0.9665 | 0.9499 | 0.9837 | 0.9918 | 0.9569 ± 0.003 |
| Logistic Regression *(baseline)* | 0.9273 | 0.8678 | 0.9956 | 0.9999 | — |
| KNN | 0.9791 | 1.0000 | 0.9590 | 0.9933 | — |
| SVM | 0.9617 | 0.9503 | 0.9733 | 0.9996 | — |

**Modèle retenu pour la production** : Random Forest (F1 = 0.9865, AUC-ROC = 1.0000, stabilité CV excellente).

**Interprétation** : Le gain apporté par le Random Forest par rapport à la baseline linéaire est de **+7 points de F1**. La faible variance cross-validation (±0.003) confirme l'absence de surapprentissage.

### 2.3 Résultats — Tâche 2 : prédiction préventive (`panne_dans_24h`)

**Contexte** : 2,46% de positifs — tâche plus difficile car elle prédit l'avenir sans historique temporel en temps réel.

| Algorithme | F1 | Precision | Recall | AUC-ROC | CV F1 (5-fold) |
|------------|-----|-----------|--------|---------|----------------|
| Random Forest | 0.8407 | 0.7294 | 0.9922 | 0.9993 | 0.8392 ± 0.004 |
| **Gradient Boosting** | **0.9635** | 0.9520 | 0.9752 | 0.9998 | 0.9605 ± 0.004 |
| Decision Tree | 0.7385 | 0.5886 | 0.9907 | 0.9893 | 0.7383 ± 0.006 |
| Logistic Regression *(baseline)* | 0.6176 | 0.4490 | 0.9891 | 0.9961 | — |
| KNN | 0.9557 | 0.9165 | 0.9984 | 0.9996 | — |
| SVM | 0.7819 | 0.6423 | 0.9990 | 0.9988 | — |

**Modèle retenu pour la production** : Gradient Boosting (F1 = 0.9635).

**Interprétation** : Le gain par rapport à la baseline est de **+34,6 points de F1**. Le Recall de 97,52% signifie que seulement 2,48% des pannes imminentes ne sont pas détectées — niveau acceptable pour un usage industriel.

### 2.4 Résultats — Tâche 3 : estimation RUL (`rul_heures`)

| Algorithme | MAE | RMSE | R² |
|------------|-----|------|----|
| **Random Forest** | **8,16 h** | 12,23 h | **0,9974** |
| Gradient Boosting | 11,49 h | 15,97 h | 0,9956 |

**Modèle retenu** : Random Forest (MAE = 8,16h, R² = 0,9974).

**Interprétation** : L'erreur moyenne de 8 heures est **largement suffisante** pour planifier une intervention de maintenance (les équipes disposent d'une fenêtre d'action de 24h). Le R² de 0,9974 confirme que le modèle explique 99,74% de la variance du RUL.

### 2.5 Tests unitaires automatisés

23 tests unitaires répartis dans 3 fichiers pytest :

| Fichier | Nombre de tests | Couverture |
|---------|----------------|------------|
| `tests/test_api.py` | 7 | Endpoints HTTP, réponses valides/invalides, codes d'erreur 422 |
| `tests/test_model.py` | 8 | Chargement des modèles, formats de prédiction, cohérence des sorties |
| `tests/test_data_preparation.py` | 8 | Pipeline ETL, feature engineering, encodage catégoriel |
| **Total** | **23** | **Pipeline complet** |

**Résultat** : tous les tests passent (`pytest tests/ -v` — 23 passed, 0 failed).

Ces tests sont exécutés automatiquement à chaque commit via le pipeline CI/CD GitHub Actions (Python 3.11 et 3.12).

---

## 3. Démarche de validation fonctionnelle

### 3.1 Critères de validation fonctionnelle

| Critère | Méthode de test | Résultat |
|---------|----------------|----------|
| Prédiction en temps réel depuis le dashboard | Saisie manuelle de valeurs capteurs | ✅ Résultat instantané avec jauge de risque et recommandation |
| Upload CSV de plusieurs machines | Chargement d'un fichier multi-lignes | ✅ Prédictions en batch, tableau de résultats téléchargeable |
| Niveaux d'alerte cohérents | Tests avec probabilités dans chaque plage (<30%, 30–50%, 50–80%, >80%) | ✅ Couleurs et recommandations correctement attribuées |
| Réponse API avec entrées invalides | Requête POST avec champs manquants | ✅ Erreur HTTP 422 avec message explicite |
| Documentation Swagger auto-générée | Accès à `/docs` | ✅ Interface OpenAPI fonctionnelle |
| Démarrage Docker en une commande | `docker-compose up --build` | ✅ API et Dashboard opérationnels en < 2 min |

### 3.2 Seuils d'alerte calibrés

Les seuils d'alerte ont été calibrés sur la distribution des probabilités de sortie des modèles sur le jeu de test :

| Niveau | Seuil de probabilité | Couleur | Action recommandée |
|--------|---------------------|---------|-------------------|
| Normal | < 30% | Vert | Aucune action |
| Surveillance | 30–50% | Jaune | Surveillance renforcée |
| Maintenance préventive | 50–80% | Orange | Intervention sous 24h |
| Critique | > 80% | Rouge | Arrêt immédiat recommandé |

---

## 4. Limites identifiées de la solution

| Limite | Impact | Mitigation envisageable |
|--------|--------|------------------------|
| **Données synthétiques** : les corrélations sont linéarisées, les pannes multi-causes ne sont pas modélisées | Les modèles pourraient sous-performer sur des modes de défaillance réels non représentés | Réentraînement sur des données réelles dès la phase pilote |
| **Déséquilibre des classes** (~2,5% de positifs) : un changement de la fréquence réelle des pannes dégraderait les performances | Dégradation possible du F1 si le taux de pannes diffère significativement | Réévaluation des seuils et réentraînement périodique |
| **Historique 24h requis** : les features glissantes nécessitent au moins 24 relevés horaires consécutifs par machine | Indisponibilité de prédiction pour les machines nouvellement raccordées ou après interruption du flux de données | Buffer de données côté collecte, alerte dédiée si historique insuffisant |
| **Pas de streaming temps réel** : le prototype ingère des fichiers CSV, pas de flux Kafka | Latence de prédiction potentiellement inadaptée à des machines à dégradation rapide | Intégration Apache Kafka prévue en V2 |
| **Monolinguisme** : interface uniquement en français | Friction pour les sites espagnols | Internationalisation (i18n) prévue avant déploiement phase 3 |

---

## 5. Projection de mise en œuvre dans l'environnement industriel MECHA

### 5.1 Conditions de déploiement

Le déploiement de la solution suppose les prérequis techniques et organisationnels suivants :

**Prérequis techniques :**
- Docker Engine ≥ 20.10 et docker-compose ≥ 1.29 sur chaque serveur de site
- Serveur Linux (recommandé) ou Windows Server avec WSL2
- RAM minimale : 4 Go par service (8 Go recommandés) ; CPU : 4 cœurs
- Connectivité réseau entre le serveur de déploiement et les capteurs IoT (protocole MQTT ou HTTP)
- Port 8000 (API) et 8501 (Dashboard) ouverts sur le réseau local du site

**Prérequis organisationnels :**
- Désignation d'un **référent technique** par site (formé à Docker et à l'administration de l'API)
- Validation par la DSI de la politique de sécurité réseau (authentification, TLS)
- Consultation du DPO avant traitement de données de production (logs d'intervention nominatifs)
- Intégration dans les processus de planification maintenance existants

### 5.2 Stratégie de déploiement progressif

| Étape | Période | Périmètre | Mode |
|-------|---------|-----------|------|
| **Phase 1 — Pilote** | Mois 1–2 | 1 site France, 10–20 machines | Observation (alertes générées, non intégrées à la planification) |
| **Phase 2 — Extension France** | Mois 3–4 | 3 sites France, totalité des machines | Opérationnel (alertes intégrées à la planification maintenance) |
| **Phase 3 — International** | Mois 5–6 | 2 sites Espagne | Opérationnel avec adaptation linguistique |

**Critères de passage de phase :**
- Taux d'adoption du dashboard > 80% des techniciens du site
- Moins de 5 incidents techniques non résolus par semaine
- Retour positif des responsables maintenance sur la pertinence des alertes
- Aucun arrêt de production causé par un faux négatif (panne non détectée)

### 5.3 Impacts techniques

| Impact | Description | Niveau |
|--------|-------------|--------|
| **Infrastructure réseau** | Ajout de 2 services Docker par site (API + Dashboard) | Faible — containerisé, sans impact sur l'infrastructure existante |
| **Collecte des données capteurs** | Mise en place d'un flux de données horaire vers `data/raw/` | Moyen — nécessite un connecteur IoT→CSV ou IoT→API |
| **Intégration SCADA/MES** | Les alertes de l'API peuvent être consommées par le SCADA via webhook | Moyen — dépend du SCADA en place (à évaluer site par site) |
| **Réentraînement périodique** | Les modèles doivent être réentraînés tous les 3–6 mois avec de nouvelles données | Moyen — nécessite un pipeline MLOps ou une procédure manuelle |
| **Surveillance opérationnelle** | Monitoring de l'API (health check, logs) à intégrer dans les outils de supervision DSI | Faible — health check déjà disponible sur `/health` |

### 5.4 Impacts organisationnels

| Acteur | Impact | Action requise |
|--------|--------|----------------|
| **Techniciens maintenance** | Modification de la routine quotidienne — consultation du dashboard en début de poste | Formation (2h) + accompagnement terrain 2 semaines |
| **Responsables maintenance** | Évolution du processus de planification des interventions | Formation (½ journée) + adaptation du processus de gestion des ordres de travail |
| **DSI** | Administration de 2 nouveaux services Docker par site | Formation technique (1 journée) + documentation d'exploitation |
| **Direction industrielle** | Suivi de nouveaux KPI (taux de faux positifs, réduction des arrêts) | Intégration dans les tableaux de bord de direction |

### 5.5 Points de vigilance

| Point de vigilance | Description | Mesure préventive |
|-------------------|-------------|------------------|
| **Dérive des modèles** | Les performances des modèles peuvent se dégrader si les conditions de fonctionnement des machines évoluent (nouvelles machines, changements de procédés) | Surveillance mensuelle des métriques en production ; réentraînement tous les 3–6 mois |
| **Qualité des données en entrée** | Des capteurs défaillants ou mal calibrés produiront des prédictions erronées | Contrôles de plausibilité à l'ingestion (valeurs hors limites physiques → alerte) |
| **Faux positifs répétés** | Des alertes "Critique" non suivies d'une panne réelle éroderont la confiance des techniciens | Phase pilote en observation pour calibrer les seuils avant passage en opérationnel |
| **Dépendance à l'infrastructure Docker** | Une panne du serveur de déploiement rend les prédictions indisponibles | Plan de continuité : basculement manuel vers la planification préventive classique |
| **RGPD** | En production, les logs d'intervention peuvent contenir des données personnelles (nom du technicien) | Pseudonymisation des logs avant ingestion ; consultation du DPO avant déploiement avec données réelles |

---

## 6. Perspectives d'évolution (V2)

| Évolution | Justification | Priorité |
|-----------|--------------|----------|
| **Streaming temps réel (Apache Kafka)** | Réduire la latence des prédictions de 1h à quelques secondes | Haute |
| **Réentraînement automatique (MLOps)** | Maintenir les performances des modèles sans intervention manuelle | Haute |
| **Explicabilité SHAP** | Fournir aux techniciens la raison de chaque alerte (quelle feature a déclenché l'alerte) | Moyenne |
| **Adaptation multi-sites (transfer learning)** | Spécialiser les modèles par type de machine ou par site | Moyenne |
| **Alerting intégré (email, SMS, SCADA)** | Notifier automatiquement les techniciens sans consultation du dashboard | Moyenne |
| **Interface multilingue (FR/ES)** | Rendre la solution accessible aux équipes espagnoles | Haute (avant phase 3) |

---

## 7. Conclusion

La solution MECHA atteint les objectifs techniques et fonctionnels définis dans le cahier des charges :

- **F1-Score de 0,9865** sur la détection de panne active, **0,9635** sur la prédiction préventive 24h
- **MAE de 8,16 heures** sur le RUL — précision suffisante pour planifier des interventions
- **23 tests unitaires** passant tous, pipeline CI/CD opérationnel
- Solution **containerisée et déployable** en une commande (`docker-compose up --build`)

La phase pilote sur un site France permettra de confirmer ces performances sur des données réelles et d'affiner les seuils d'alerte avant généralisation à l'ensemble des 5 sites MECHA.
