# Rapport de Validation et Projection de Déploiement Industriel
## MECHA — Solution de Maintenance Prédictive par IA

*MSPR TPRE841 — Bloc 4 — EPSI 2025-2026*

Tous les chiffres de ce rapport proviennent de `models/training_results.json` (dernier entraînement) et de `data/processed/rapport_qualite.json`. Ils sont régénérés à chaque exécution de `python src/model_training.py`.

---

## 1. Objectifs de la validation

La validation poursuit deux axes complémentaires :

1. **Validation technique** : les modèles reconnaissent-ils réellement une machine qui va tomber en panne, et avec quelle fiabilité dans les conditions réelles de l'atelier (pannes rares) ? Les composants logiciels (préparation des données, API, alertes, dashboard) fonctionnent-ils de manière fiable ?
2. **Validation fonctionnelle** : la règle d'alerte répond-elle aux besoins exprimés lors de l'entretien client (anticiper les pannes, détecter au moins 80 % des pannes — BM1, limiter les fausses alertes critiques — BM2) avec un volume d'alertes soutenable pour les équipes ?

---

## 2. Données utilisées pour la validation

Le dataset est synthétique (`reference/generate_dataset.py`) : 30 machines, 5 usines, relevés horaires. Les sites espagnols n'ont que 12 mois d'historique (contre 18 en France), comme indiqué lors de l'entretien. Le fichier brut contient des défauts capteurs volontairement injectés ; ils sont corrigés par `src/data_preparation.py` selon les règles de `config/regles_metier.yaml` :

| Étape de nettoyage | Volume |
|---|---|
| Lignes brutes | 354 162 |
| Doublons de transmission supprimés | 354 |
| Valeurs manquantes dans le fichier brut | 15 553 |
| Valeurs hors bornes physiques (remplacées par une valeur manquante) | 829 |
| Valeurs de capteur figé (même valeur ≥ 6 h, remplacées) | 1 610 |
| Valeurs comblées (trou ≤ 3 h, dernière valeur connue) | 13 574 |
| Lignes écartées (trou > 3 h) | 2 203 (0,6 %) |
| **Lignes exploitées** | **351 605** |

Taux de relevés « panne dans les 24 h » : 2,32 % ; taux de relevés « en panne » : 2,45 %.

---

## 3. Validation technique des modèles

### 3.1 Protocole

| Tâche | Découpage train / test | Évaluation |
|---|---|---|
| Classification « panne dans 24 h » et « en panne » | Aléatoire stratifié 80 / 20 | **(a)** test rééquilibré 50/50 **et (b)** test à la distribution réelle |
| Régression RUL (heures avant panne) | **Temporel** : train avant le 23/03/2025, test après | MAE, RMSE, R² ; références naïve et linéaire |
| Détection d'anomalies (Isolation Forest) | Entraînement sur les relevés « normal » uniquement | Part des relevés en panne signalés, faux positifs sur l'état normal |
| Règle d'alerte | Temporel, rejeu heure par heure (section 4) | Comptage par événement de panne |

**Pourquoi deux évaluations pour la classification ?**

- **Test 50/50 (demandé par l'encadrant)** : avec 2,3 % de pannes, un modèle qui répond toujours « pas de panne » obtient 97,7 % d'*accuracy* sans rien détecter. Sur un test équilibré, ce même modèle plafonne à 50 % : un score élevé prouve que le modèle reconnaît vraiment une panne. C'est l'évaluation utilisée pour **comparer les modèles**.
- **Test à la distribution réelle** : il mesure ce que verra la maintenance en production, en particulier la **précision** (part des alertes qui correspondent à une vraie panne). C'est l'évaluation utilisée pour **régler les seuils et la règle d'alerte**.

Vocabulaire : l'*accuracy* (exactitude) est la part de prédictions correctes ; la **précision** est la part des alertes justifiées ; le **rappel** est la part des pannes détectées.

Le train est rééquilibré 50/50 dans les deux cas. Une validation croisée 5 plis est appliquée aux trois modèles à base d'arbres. Un modèle naïf (« jamais de panne » pour la classification, « RUL moyen » pour la régression) sert de référence plancher.

### 3.2 Classification « panne dans les 24 h » (modèle des alertes)

**(a) Test 50/50 — capacité à reconnaître une panne**

| Modèle | Accuracy | Précision | Rappel | F1 | AUC-ROC | F1 validation croisée |
|---|---|---|---|---|---|---|
| Modèle naïf (référence) | 0,500 | 0,000 | 0,000 | 0,000 | 0,500 | — |
| **Random Forest** (déployé) | 0,874 | 0,830 | 0,940 | 0,882 | 0,940 | 0,881 ± 0,008 |
| Gradient Boosting | 0,900 | 0,865 | 0,949 | **0,905** | **0,959** | 0,899 ± 0,011 |
| Decision Tree | 0,836 | 0,810 | 0,877 | 0,842 | 0,870 | 0,839 ± 0,005 |
| Régression logistique | 0,848 | 0,817 | 0,896 | 0,855 | 0,912 | — |
| KNN | 0,849 | 0,813 | 0,907 | 0,858 | 0,905 | — |
| SVM | 0,864 | 0,813 | 0,945 | 0,874 | 0,932 | — |

**(b) Distribution réelle — ce que verra la maintenance** (70 322 relevés de test, 2,32 % de pannes sous 24 h)

| Modèle | Accuracy | Précision | Rappel | F1 | AUC-ROC | Fausses alertes au niveau « critique » |
|---|---|---|---|---|---|---|
| Modèle naïf (référence) | **0,977** | 0,000 | 0,000 | 0,000 | 0,500 | aucune alerte |
| **Random Forest** (déployé) | 0,824 | 0,111 | 0,940 | 0,198 | 0,941 | 79,3 % |
| Gradient Boosting | 0,862 | 0,138 | 0,949 | 0,242 | 0,959 | 76,5 % |
| Decision Tree | 0,810 | 0,098 | 0,877 | 0,176 | 0,878 | 88,1 % |
| Régression logistique | 0,812 | 0,101 | 0,896 | 0,181 | 0,912 | 85,1 % |
| KNN | 0,810 | 0,101 | 0,907 | 0,181 | 0,911 | 86,4 % |
| SVM | 0,801 | 0,100 | 0,945 | 0,180 | 0,934 | 82,9 % |

Répartition des relevés de test du Random Forest déployé par niveau de risque (seuils de `config/regles_metier.yaml`) :

| Niveau | Relevés | Dont vraies pannes sous 24 h | Part de fausses alertes |
|---|---|---|---|
| Critique (≥ 80 %) | 4 957 | 1 026 | 79,3 % |
| Élevé (50–80 %) | 8 882 | 507 | 94,3 % |
| Moyen (30–50 %) | 6 290 | 72 | 98,9 % |
| Pannes classées « faible » (non signalées) | — | 25 sur 1 630 | — |

**Interprétation**

- Sur le test 50/50, tous les modèles dépassent largement le modèle naïf (F1 de 0,84 à 0,91 contre 0) : ils reconnaissent réellement les situations de pré-panne. Le **Gradient Boosting** est le meilleur sur toutes les métriques ; le Random Forest, actuellement déployé par l'API, est un cran en dessous (voir section 7).
- Sur la distribution réelle, l'*accuracy* du modèle naïf (97,7 %) est supérieure à celle de tous les modèles : c'est le paradoxe que le test 50/50 permet d'éviter.
- Le rappel reste élevé (94 % pour le Random Forest) mais la **précision relevé par relevé est faible (11 %)** : raisonner relevé par relevé produirait trop de fausses alertes. D'où la règle d'alerte validée en section 4, qui raisonne par événement et exige plusieurs relevés consécutifs.

### 3.3 Classification « machine en panne »

| Modèle | Test 50/50 : F1 | AUC-ROC | F1 validation croisée | Réel : précision | Réel : rappel | Réel : F1 |
|---|---|---|---|---|---|---|
| Modèle naïf | 0,000 | 0,500 | — | 0,000 | 0,000 | 0,000 (accuracy 0,975) |
| **Random Forest** | 0,922 | 0,977 | 0,921 ± 0,002 | 0,194 | 0,945 | 0,322 |
| Gradient Boosting | **0,935** | **0,986** | 0,936 ± 0,002 | 0,231 | 0,952 | 0,372 |
| Decision Tree | 0,889 | 0,941 | 0,890 ± 0,006 | 0,180 | 0,885 | 0,300 |
| Régression logistique | 0,888 | 0,960 | — | 0,153 | 0,903 | 0,262 |
| KNN | 0,904 | 0,964 | — | 0,219 | 0,897 | 0,352 |
| SVM | 0,925 | 0,979 | — | 0,202 | 0,942 | 0,333 |

En production, l'état « en panne » est déjà connu du MES / SCADA : ce modèle sert surtout de contrôle de cohérence et de démonstration. Les alertes reposent sur le modèle « panne dans 24 h ».

### 3.4 Régression : temps restant avant panne (RUL)

Découpage temporel : entraînement sur les relevés antérieurs au 23/03/2025, test sur la période suivante. Relevés « en panne » exclus.

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| Modèle naïf (RUL moyen) | 223,7 h | 248,5 h | 0,000 |
| Régression linéaire (référence) | 105,3 h | 134,4 h | **0,708** |
| **Random Forest** (déployé) | 105,5 h | 155,8 h | 0,607 |
| Gradient Boosting | **101,5 h** | 147,3 h | 0,649 |

**Interprétation**

- Les modèles divisent par deux l'erreur du modèle naïf : le RUL apporte une vraie information de **priorisation** (une machine à « ~100 h » est plus urgente qu'une machine à « ~600 h »).
- Une erreur moyenne d'environ 100 h interdit de l'utiliser pour planifier à l'heure près. Il est présenté comme un **ordre de grandeur** dans le dashboard et ne déclenche aucune alerte.
- Les modèles d'arbres ne font pas mieux que la régression linéaire : avec la seule mesure instantanée, la relation exploitable est essentiellement linéaire (dérive progressive des capteurs). Un découpage aléatoire donnait une erreur de ~61 h : il surestimait la performance car des heures voisines d'une même machine se retrouvaient dans le train et le test.

### 3.5 Détection d'anomalies (Isolation Forest)

| Indicateur | Valeur |
|---|---|
| Relevés d'entraînement (état normal) | 286 220 |
| Contamination supposée | 5 % |
| Relevés « en panne » signalés comme anomalie | 83,1 % |
| Relevés « normaux » signalés (faux positifs) | 5,0 % (fixé par la contamination) |

L'Isolation Forest est complémentaire : il ne nécessite aucun historique de pannes et signale des comportements capteurs jamais vus (nouveau mode de défaillance, capteur déréglé). Une alerte « anomalie » n'est émise qu'au niveau le plus fort, sur 3 relevés consécutifs.

---

## 4. Validation fonctionnelle de la règle d'alerte

### 4.1 Protocole

Pour se rapprocher de l'exploitation réelle :

1. Le Random Forest « panne dans 24 h » (mêmes paramètres que le modèle déployé) est entraîné uniquement sur les relevés antérieurs au **23/03/2025**.
2. Il est **rejoué heure par heure**, machine par machine, sur la période suivante : 30 machines, 419 machine-semaines, **77 débuts de panne**.
3. Chaque règle candidate (seuil de probabilité × nombre de relevés consécutifs) est appliquée. Une alerte bloque les suivantes de la même machine pendant 24 h (approximation de « alerte ouverte jusqu'à acquittement »). Les heures où la machine est déjà en panne sont ignorées.
4. **Comptage par événement** : une panne est *anticipée* si une alerte a été émise dans les 24 h précédant son début ; une alerte est *suivie d'une panne* si une panne commence dans l'horizon considéré (24 h, 72 h ou 7 jours).

### 4.2 Résultats

| Seuil | Relevés consécutifs | Alertes | Pannes anticipées | Suivies d'une panne sous 24 h | sous 72 h | sous 7 j | Alertes / machine / semaine | Anticipation médiane |
|---|---|---|---|---|---|---|---|---|
| 50 % | 1 | 1 012 | 97,4 % | 7,5 % | 22,2 % | 48,4 % | 2,42 | 12 h |
| 50 % | 2 | 736 | 97,4 % | 10,2 % | 29,8 % | 61,8 % | 1,76 | 13 h |
| 50 % | 3 | 574 | 90,9 % | 12,2 % | 36,5 % | 73,4 % | 1,37 | 13 h |
| 50 % | 4 | 477 | 92,2 % | 14,9 % | 42,1 % | 78,8 % | 1,14 | 15 h |
| 50 % | 6 | 370 | 85,7 % | 17,8 % | 49,2 % | 84,5 % | 0,88 | 16 h |
| 80 % | 1 | 561 | 93,5 % | 12,9 % | 35,9 % | 67,8 % | 1,34 | 13 h |
| 80 % | 2 | 324 | 81,8 % | 19,4 % | 53,3 % | 85,8 % | 0,77 | 12 h |
| **80 % (critique)** | **3** | **223** | **75,3 %** | **26,0 %** | **61,3 %** | **90,3 %** | **0,53** | **14 h** |
| 80 % | 4 | 162 | 62,3 % | 29,6 % | 68,5 % | 93,6 % | 0,39 | 13 h |
| 80 % | 6 | 89 | 39,0 % | 33,7 % | 79,8 % | 98,8 % | 0,21 | 15 h |

### 4.3 Règle retenue et justification

**Règle configurée** (`config/regles_metier.yaml`, section `alertes.panne24h`) : alerte quand **3 relevés consécutifs** d'une machine sont au niveau **critique** (probabilité ≥ 80 %), sans nouvelle alerte tant que la précédente n'est pas acquittée.

C'est le meilleur compromis mesuré :

- elle reste au-dessus du besoin **BM1** à l'échelle d'une fenêtre d'intervention (75 % des pannes signalées dans les 24 h qui précèdent ; 82 % à 97 % pour les règles plus permissives, au prix de 1,5 à 4,5 fois plus d'alertes) ;
- elle divise par 4,5 le volume d'alertes par rapport à une alerte sur chaque relevé ≥ 50 % (0,53 contre 2,42 alertes par machine et par semaine, soit environ 16 alertes par semaine pour 30 machines) ;
- **9 alertes sur 10 sont suivies d'une vraie panne dans les 7 jours**, 6 sur 10 dans les 72 h.

Les règles à 4 ou 6 relevés réduisent encore les alertes mais laissent passer trop de pannes (62 % et 39 % anticipées).

### 4.4 Lecture honnête : le besoin BM2 n'est pas atteint à 24 h

Le besoin **BM2** (« moins de 5 % de fausses alertes critiques ») n'est **pas atteint** si l'on exige que la panne survienne dans les 24 h : seules **26 %** des alertes sont suivies d'une panne dans ce délai.

Ces alertes ne sont pas pour autant infondées : **90 % d'entre elles précèdent une panne dans la semaine**. Le modèle reconnaît fiablement une **machine qui se dégrade**, mais pas le **moment précis** de la panne. Deux raisons :

- **mesure instantanée** : le modèle ne voit qu'un relevé horaire, sans tendance ; le bruit capteur masque la différence entre « 24 h avant » et « 4 jours avant » ;
- **dégradation progressive** : dans les données, la dérive des capteurs commence plusieurs jours avant la panne (états « à risque » puis « critique »), ce qui reflète le comportement réel d'une usure mécanique.

**Recommandations à arbitrer avec le client :**

1. **Redéfinir l'alerte** comme « machine à inspecter, intervention à planifier sous 7 jours ». Avec cet horizon, la part d'alertes non suivies d'une panne tombe à **10 %** pour la règle retenue (et sous 5 % pour la règle à 6 relevés, au prix d'un rappel trop faible).
2. **Ajouter des features temporelles** (moyennes et écarts-types glissants sur 24 h, pente sur 72 h), calculables en production à partir de l'historique stocké dans PostgreSQL. C'est le principal levier pour gagner en précision temporelle.
3. **Calibrer pendant le pilote** les seuils et le nombre de relevés consécutifs avec les retours terrain : ils sont modifiables dans `config/regles_metier.yaml` sans toucher au code.

---

## 5. Tests automatisés

**88 tests** pytest, exécutés à chaque push par GitHub Actions (Python 3.11 et 3.12), qui construit ensuite l'image Docker et vérifie qu'elle importe correctement l'API et la couche base de données.

| Fichier | Tests | Couverture |
|---|---|---|
| `tests/test_api.py` | 28 | Prédictions **réellement exécutées** avec de petits modèles de test (panne, panne 24 h, RUL, anomalie, explication SHAP, batch) ; validation des entrées (champs manquants, bornes physiques → HTTP 422) ; webhook avec base en mémoire ; déclenchement d'alerte après N relevés consécutifs, absence d'alerte sur pic isolé, anti-doublon tant que l'alerte est ouverte, alertes séparées par machine, alerte de source « anomalie », notification externe, listing et acquittement, monitoring, comportement sans base (HTTP 503) |
| `tests/test_regles.py` | 25 | Fichier `config/regles_metier.yaml` : niveaux de risque panne et RUL aux bornes, normalisation du score d'anomalie, règle de relevés consécutifs, rejet des configurations incohérentes, chargement d'un autre fichier de règles |
| `tests/test_model.py` | 19 | Features, rééquilibrage 50/50, paradoxe de l'accuracy du modèle naïf, répartition des alertes par niveau, détection des débuts de panne, simulation de la règle d'alerte (relevés consécutifs, délai entre alertes, machine déjà en panne), comptage par événement |
| `tests/test_data_preparation.py` | 16 | Features glissantes, encodage, nettoyage : doublons, hors bornes, capteur figé (et répétition courte conservée), comblement limité à 3 h, trou long écarté, pas d'imputation d'une machine à l'autre, labels intacts |
| **Total** | **88** | |

---

## 6. Plan de correction : anomalies identifiées et traitées

Un audit de la solution par rapport au cahier des charges a été mené avant la soutenance. Anomalies relevées, corrections apportées et statut :

| # | Anomalie identifiée | Gravité | Correction apportée | Statut |
|---|---|---|---|---|
| 1 | CI en échec sur tous les pushs : `COPY models/` dans le Dockerfile alors que les modèles ne sont pas versionnés | Bloquante | Image sans données ni modèles (volumes) ; service `trainer` qui prépare les données et entraîne les modèles s'ils manquent ; `.dockerignore` ; vérification de l'image dans la CI | Corrigé |
| 2 | Tests API sans modèles chargés : toutes les prédictions renvoyaient 503, accepté par les tests | Majeure | Modèles de test injectés, base en mémoire ; 28 tests API qui exécutent réellement les prédictions | Corrigé |
| 3 | Endpoint d'explication SHAP en erreur 500 avec shap ≥ 0.45 (format de sortie modifié) | Majeure | Prise en charge des deux formats ; explainer mis en cache ; explication du modèle 24 h ajoutée | Corrigé |
| 4 | SQLAlchemy 2.1 sélectionne le pilote psycopg v3 : base désactivée dans une image reconstruite (webhook et monitoring en 503) | Majeure | Pilote psycopg2 imposé dans l'URL de connexion ; contrôle dans la CI | Corrigé |
| 5 | Métriques publiées uniquement sur le test 50/50 : performance en production non mesurée | Majeure | Évaluation complémentaire sur la distribution réelle, taux de fausses alertes par niveau, courbes précision-rappel | Corrigé |
| 6 | RUL évalué avec un découpage aléatoire : performance surestimée (MAE ~61 h annoncée, ~105 h en réalité) | Majeure | Découpage temporel ; références naïve et linéaire | Corrigé |
| 7 | Seuils d'alerte codés en dur à 5 endroits (API, dashboard) | Moyenne | Fichier unique `config/regles_metier.yaml`, validé au démarrage et en CI | Corrigé |
| 8 | Aucune alerte ni notification : simple code couleur | Majeure | Table `alertes`, règle de relevés consécutifs, anti-doublon, acquittement, notification par webhook configurable | Corrigé |
| 9 | Dashboard chargeant directement les modèles au lieu de passer par l'API | Moyenne | Toutes les prédictions passent par l'API (dont un endpoint batch pour l'import CSV) | Corrigé |
| 10 | Pas d'horodatage de la mesure : seule l'heure de réception était stockée | Moyenne | Champ `measured_at` transmis par le simulateur et stocké | Corrigé |
| 11 | Données sans défauts de qualité, contrairement au contexte décrit (bruit, valeurs manquantes, hétérogénéité) | Moyenne | Fichier brut avec défauts injectés, historique plus court en Espagne, étape de nettoyage paramétrée et bilan qualité | Corrigé |
| 12 | Notebook d'exploration portant sur les anciens datasets publics | Mineure | Ancien notebook et datasets obsolètes retirés du dépôt ; nouveau notebook `notebooks/exploration_mecha.ipynb` sur le dataset MECHA | Corrigé |
| 13 | Documentation incohérente : trois jeux de métriques différents selon les documents | Majeure | Documentation réalignée sur `training_results.json` | Corrigé |
| 14 | Besoin BM2 non atteint à 24 h (section 4.4) | Majeure | Mesuré et documenté ; recommandations (horizon 7 jours, features temporelles, calibration pilote) | **Ouvert — à arbitrer avec le client** |
| 15 | Stack `docker compose` complète non exécutée de bout en bout lors de l'audit (Docker Desktop arrêté sur le poste) | Moyenne | Chaîne vérifiée hors Docker sur un vrai PostgreSQL 18 : mise à niveau automatique d'une base à l'ancien schéma (table `alertes`, colonne `measured_at`), webhook, déclenchement / anti-doublon / acquittement d'alerte, monitoring, et simulateur sur données brutes (258 relevés acceptés, 3 rejetés hors bornes, 9 non transmis, 0 erreur) ; 5 pages du dashboard exécutées sans erreur contre l'API | **Ouvert — lancer `docker compose up --build` avant la soutenance** |

---

## 7. Limites identifiées

| Limite | Impact | Mitigation envisageable |
|---|---|---|
| **Données synthétiques** : corrélations simplifiées, pas de pannes multi-causes, taux de panne élevé (~2,5 % du temps) | Les performances réelles peuvent différer | Ré-entraînement sur les données GMAO / SCADA dès le pilote |
| **Mesure instantanée uniquement** | Précision temporelle des alertes limitée (section 4.4) | Features temporelles calculées sur l'historique en base |
| **RUL imprécis** (~100 h d'erreur moyenne) | Utilisable pour prioriser, pas pour planifier à l'heure | Affiché comme ordre de grandeur ; pas d'alerte sur le RUL |
| **Modèle déployé non optimal** : l'API utilise le Random Forest alors que le Gradient Boosting est meilleur sur les deux évaluations | Quelques points de F1 et de précision perdus | Basculer l'API sur le Gradient Boosting et revalider la règle d'alerte |
| **Hyperparamètres fixés sans recherche systématique** | Performance possiblement sous-optimale | Recherche par validation croisée temporelle (GridSearch / RandomizedSearch) |
| **Classification évaluée par découpage aléatoire** | Peu d'impact mesuré (le rejeu temporel de la section 4 donne des résultats cohérents) | Généraliser le découpage temporel |
| **Pas d'authentification sur l'API ni le dashboard** | Inacceptable en production | Reverse proxy TLS + authentification (annuaire d'entreprise) |
| **Interface uniquement en français** | Friction pour les sites espagnols | Internationalisation avant la phase 3 |

---

## 8. Projection de mise en œuvre dans l'environnement industriel MECHA

### 8.1 Architecture de déploiement

La solution est livrée sous forme de 5 services Docker (`docker-compose.yml`) :

| Service | Rôle | En production |
|---|---|---|
| `postgres` | Stockage des relevés, prédictions et alertes | Base du site ou base centrale (entrepôt) |
| `trainer` | Prépare les données et entraîne les modèles s'ils sont absents, puis s'arrête | Exécuté à l'installation et à chaque ré-entraînement |
| `api` | Inférence, règles d'alerte, notification, endpoints de monitoring | Service permanent |
| `dashboard` | Interface métier (5 pages) | Service permanent |
| `simulator` | Rejoue le fichier brut vers le webhook | **Remplacé** par un connecteur SCADA / MES qui appelle `POST /webhook/sensor` |

Flux de production : capteurs → SCADA / MES → connecteur → `POST /webhook/sensor` (relevé horodaté) → contrôle de plausibilité → inférence → PostgreSQL → règle d'alerte → notification (webhook vers l'outil de ticketing ou de messagerie) et dashboard.

### 8.2 Conditions de déploiement

**Prérequis techniques :**
- Docker Engine avec Docker Compose v2 (`docker compose`) sur un serveur par site, ou un serveur central si le réseau inter-sites le permet
- Serveur Linux recommandé ; 4 cœurs ; 8 Go de RAM pour l'entraînement (l'inférence seule en demande beaucoup moins)
- Espace disque : environ 0,5 Go pour les modèles et données préparées, plus la croissance de la base (un relevé et une prédiction par machine et par heure)
- Connecteur SCADA / MES capable d'émettre une requête HTTP par relevé (ou un relais MQTT → HTTP)
- Ports 8000 (API) et 8501 (dashboard) ouverts sur le réseau local du site uniquement ; port 5432 non exposé hors du serveur

**Prérequis organisationnels :**
- Un **référent technique** par site (Docker, administration de l'API, fichier de règles)
- Validation par la DSI de la politique de sécurité (authentification, TLS, comptes de service)
- Consultation du DPO avant tout traitement de données de production nominatives
- Intégration des alertes dans le processus de gestion des ordres de travail (GMAO)
- Arbitrage du client sur la définition de l'alerte (horizon 24 h ou 7 jours, section 4.4)

### 8.3 Stratégie de déploiement progressif

| Étape | Période | Périmètre | Mode |
|---|---|---|---|
| **Phase 1 — Pilote** | Mois 1–2 | 1 site France, 10–20 machines | Observation : alertes générées et acquittées, non intégrées à la planification ; calibration du fichier de règles ; ré-entraînement sur données réelles |
| **Phase 2 — Extension France** | Mois 3–4 | 3 sites France | Opérationnel : alertes intégrées à la planification maintenance |
| **Phase 3 — International** | Mois 5–6 | 2 sites Espagne | Opérationnel, interface traduite ; attention à l'historique plus court de ces sites |

**Critères de passage de phase :**
- Taux d'adoption du dashboard > 80 % des techniciens du site
- Moins de 5 incidents techniques non résolus par semaine
- Part d'alertes suivies d'une panne ou d'une anomalie constatée sur l'horizon convenu avec le client, mesurée à partir des acquittements
- Aucun arrêt majeur dû à une panne non signalée sans analyse de cause

### 8.4 Impacts techniques

| Impact | Description | Niveau |
|---|---|---|
| **Collecte des données** | Connecteur SCADA / MES → webhook ; horodatage de la mesure à la source obligatoire | Moyen — dépend du SCADA de chaque site |
| **Infrastructure** | 4 services permanents + PostgreSQL par site, ou mutualisation centrale | Faible à moyen |
| **Qualité des données** | Contrôle de plausibilité à l'ingestion (relevés hors bornes rejetés) ; nettoyage à l'entraînement | Moyen — calibrer les bornes par type de machine |
| **Ré-entraînement** | Trimestriel (`frequences.reentrainement_modeles`), par le service `trainer` avec `--force`, suivi d'une revalidation de la règle d'alerte | Moyen — procédure à formaliser (MLOps) |
| **Supervision** | `/health` (modèles chargés, base joignable) à intégrer à la supervision DSI | Faible |

### 8.5 Impacts organisationnels

| Acteur | Impact | Action requise |
|---|---|---|
| **Techniciens maintenance** | Consultation des alertes en début de poste ; acquittement à la prise en charge | Formation 2 h + accompagnement terrain 2 semaines |
| **Responsables maintenance** | Alertes intégrées aux ordres de travail ; arbitrage des interventions | Formation ½ journée ; adaptation du processus GMAO |
| **DSI** | Administration des services, connecteur SCADA, sauvegardes, sécurité | Formation technique 1 journée + documentation d'exploitation |
| **Direction industrielle** | Suivi de nouveaux indicateurs (pannes anticipées, alertes par semaine, arrêts évités) | Intégration aux tableaux de bord de direction |

### 8.6 Points de vigilance

| Point de vigilance | Description | Mesure préventive |
|---|---|---|
| **Fatigue d'alerte** | Avec une définition à 24 h, 3 alertes sur 4 ne sont pas suivies d'une panne dans ce délai | Horizon de 7 jours, relevés consécutifs, anti-doublon, calibration pendant le pilote |
| **Dérive des modèles** | Nouvelles machines, changement de procédés, vieillissement du parc | Suivi mensuel des indicateurs d'alerte ; ré-entraînement trimestriel |
| **Qualité des capteurs** | Capteurs figés, pertes de transmission, valeurs aberrantes | Bornes physiques à l'ingestion ; bilan qualité à chaque préparation ; alerte « anomalie » |
| **Hétérogénéité des sites** | Historique plus court et capteurs moins fiables en Espagne | Démarrage en mode observation ; règles par site possibles (fichier de règles distinct) |
| **Continuité de service** | Une panne du serveur rend les prédictions indisponibles | Retour à la maintenance préventive planifiée ; sauvegarde de la base |
| **Sécurité** | API et dashboard sans authentification dans le prototype | Reverse proxy TLS + authentification avant toute mise en production |
| **RGPD** | L'acquittement n'est pas nominatif dans le prototype ; il le deviendrait s'il était rattaché à un technicien | Pseudonymisation ; consultation du DPO ; durée de conservation définie |

---

## 9. Valeur pour la direction financière (estimation paramétrique)

Cette section donne une **méthode de chiffrage**, pas un résultat. Les volumes proviennent de la validation (section 4). **Les hypothèses de coût sont à valider avec la direction financière** ; le taux de panne et la durée des pannes sont ceux du simulateur et doivent être remplacés par les données GMAO.

### 9.1 Formule

> **Gain annuel net = N_pannes × R × D × p × C_arrêt − N_alertes × C_inspection − C_exploitation**

| Paramètre | Signification | Valeur | Origine |
|---|---|---|---|
| N_pannes | Pannes par an sur le parc | 287 | Mesuré : 0,184 panne / machine / semaine × 30 machines × 52 semaines |
| R | Part des pannes anticipées | 75,3 % | Mesuré : règle retenue (section 4.2) |
| N_alertes | Alertes par an | 831 | Mesuré : 0,533 alerte / machine / semaine × 30 × 52 |
| D | Durée moyenne d'un arrêt non planifié | 26 h | Paramètre du simulateur (4 à 48 h) — **à remplacer par la GMAO** |
| p | Part de l'arrêt évitée quand la panne est anticipée | 25 % à 50 % | **Hypothèse à valider** |
| C_arrêt | Coût d'une heure d'arrêt non planifié | 500 € à 1 000 € | **Hypothèse à valider** |
| C_inspection | Coût d'une inspection déclenchée par une alerte | 150 € à 300 € | **Hypothèse à valider** |
| C_exploitation | Coût annuel d'exploitation de la solution (infrastructure, support, ré-entraînement) | 50 k€ à 80 k€ | **Hypothèse à valider** |

### 9.2 Exemple chiffré

Heures d'arrêt concernées par une panne anticipée : 287 × 0,753 × 26 h ≈ **5 619 h / an**.

| Scénario | p | C_arrêt | C_inspection | C_exploitation | Arrêts évités | Gain brut | Coût des alertes | **Gain net** |
|---|---|---|---|---|---|---|---|---|
| Prudent | 25 % | 500 €/h | 150 € | 50 k€ | 1 405 h | 702 k€ | 125 k€ | **≈ 528 k€ / an** |
| Central | 50 % | 1 000 €/h | 300 € | 80 k€ | 2 809 h | 2 809 k€ | 249 k€ | **≈ 2,48 M€ / an** |

**Seuil de rentabilité** (scénario central hors C_arrêt) : la solution est rentable dès que l'heure d'arrêt non planifié coûte plus de (249 k€ + 80 k€) / 2 809 h ≈ **117 €**.

Ces montants sont élevés parce que le simulateur génère environ une panne par machine toutes les 5 à 6 semaines. Le raisonnement à présenter à la direction est la formule et le seuil de rentabilité, recalculés avec les taux de panne réels du pilote.

---

## 10. Conclusion

- **Ce qui est démontré** : les modèles reconnaissent réellement les situations de pré-panne (F1 de 0,88 à 0,91 sur le test 50/50, contre 0 pour le modèle naïf). Une règle d'alerte simple et paramétrable signale 75 % des pannes dans les 24 h qui les précèdent, avec environ une alerte toutes les deux semaines par machine, dont 9 sur 10 précèdent une panne dans la semaine. La chaîne complète (données brutes → nettoyage → modèles → API → alertes → dashboard) est testée automatiquement et conteneurisée.
- **Ce qui ne l'est pas** : la précision temporelle à 24 h (BM2 non atteint) et la précision du RUL (~100 h d'erreur), toutes deux limitées par l'usage de la seule mesure instantanée sur des données simulées.
- **Décisions attendues du client** : définition de l'horizon d'alerte (24 h ou 7 jours), lancement du pilote en mode observation, accès aux données GMAO / SCADA réelles pour ré-entraîner et chiffrer la valeur.
