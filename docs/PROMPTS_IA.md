# Utilisation de l'IA générative — prompts et intégration

*MSPR TPRE841 — Bloc 4 — EPSI 2025-2026*

Le cahier des charges (p. 5) autorise l'usage d'un outil d'IA générative, à condition de **fournir les prompts utilisés** et d'**expliquer comment les productions ont été analysées, adaptées et intégrées** dans la solution finale. Ce document répond à cette exigence.

Principes appliqués par l'équipe :
- l'IA propose, l'équipe **décide** : chaque choix structurant (méthode d'évaluation, données, périmètre des livrables) a fait l'objet d'un arbitrage explicite ;
- aucune production n'est intégrée sans **vérification par exécution** (tests automatisés, réentraînement, comparaison des métriques avant/après) ;
- les chiffres des documents proviennent des fichiers produits par le code (`models/training_results.json`, `data/processed/rapport_qualite.json`), jamais d'une affirmation de l'IA.

---

## 1. Session d'audit et de correction avec un assistant IA de code

| | |
|---|---|
| **Outil** | Claude Code (assistant IA de code d'Anthropic), dans l'éditeur VS Code |
| **Date** | 09/11/2026 |
| **Objectif** | Auditer le projet par rapport au cahier des charges, puis corriger les manques |
| **Accès de l'outil** | Lecture du dépôt, exécution de commandes locales (tests, entraînement), modification des fichiers ; aucune donnée réelle d'entreprise (dataset entièrement simulé) |

Les prompts sont reproduits **tels qu'ils ont été saisis**, dans l'ordre, fautes de frappe comprises.

### Prompt 1

> audit ce projet voir si on a tout ou si il manque des point encore maintenant

**Production de l'IA** : lecture du sujet PDF et de tout le code, puis rapport d'audit comparant chaque exigence du cahier des charges à l'existant. Constats principaux : CI en échec, tests d'API qui ne testaient aucune prédiction, documentation avec trois jeux de métriques différents, métriques mesurées uniquement sur un test rééquilibré 50/50, RUL surestimé par un découpage aléatoire, absence de paramétrage des seuils et d'alertes, données sans défauts de qualité. Les constats chiffrés ont été appuyés par des expériences exécutées (comparaison de découpages aléatoire / temporel / par machine, évaluation du modèle déployé sur la distribution réelle) et par l'état des exécutions GitHub Actions.

**Analyse par l'équipe** : la présentation du test 50/50 comme une erreur a été contestée (voir prompts 2 et 3) ; les autres constats ont été acceptés.

### Prompt 2

> la distribution 50/50 c'etait le prof qui nous avais dis de faire sa mais plus sur de la raison apres 8 mois

**Production de l'IA** : analyse de l'historique Git (le rééquilibrage a d'abord porté sur l'entraînement seul, puis sur le test, le même jour) ; explication des raisons possibles de la consigne ; proposition de **conserver** le 50/50 et d'ajouter une évaluation complémentaire sur la distribution réelle.

**Analyse par l'équipe** : la consigne de l'encadrant est maintenue ; l'IA a corrigé son audit (le 50/50 n'est pas une erreur, c'est l'évaluation complémentaire qui manquait).

### Prompt 3

> je pense que c'etait pour les test pour voir si le modele ne repond pas juste vrai tout le temps et a quand meme 97% de precision
> alors que sur des donner a 50% il doit reelement etre capable d'identifier

**Production de l'IA** : confirmation du raisonnement (paradoxe de l'*accuracy* : un modèle qui répond toujours « pas de panne » obtient ~97,7 % sur les données réelles et 50 % sur un test équilibré) ; correction du vocabulaire (*accuracy* ≠ précision) ; proposition d'une ligne « modèle naïf » dans les résultats pour rendre l'argument visible.

**Analyse par l'équipe** : justification retenue telle quelle pour la soutenance ; vocabulaire corrigé dans les documents.

### Prompt 4

> ok vasy fais

**Production de l'IA** : ajout dans `src/model_training.py` du modèle naïf de référence, de l'évaluation sur la distribution réelle (précision, rappel, taux de fausses alertes par niveau de risque) et des courbes précision-rappel ; 5 tests unitaires.

**Vérification** : tests exécutés (29 réussis à ce stade) ; réentraînement complet puis comparaison automatique avec les métriques d'avant : métriques 50/50 **identiques** (écart < 10⁻¹²), ce qui prouve que les modèles n'ont pas changé.

### Prompt 5

> aussi on est d'accord les donner de test ne sont pas les meme que celle d'entrainement

**Production de l'IA** : vérification par script sur les données réelles du projet : aucune ligne commune entre train et test (y compris pour les jeux 50/50), aucun relevé capteur identique des deux côtés ; rappel de la nuance (heures voisines d'une même machine dans le découpage aléatoire, sans effet mesuré sur la classification mais avec effet sur le RUL).

**Analyse par l'équipe** : réponse acceptée ; le découpage temporel du RUL est retenu.

### Prompt 6

> ok et pour tout le reste a corriger c'est bon

**Production de l'IA** : avant d'agir, l'IA a posé trois questions de décision à l'équipe (voir ci-dessous), puis a réalisé les corrections : fichier de règles métier `config/regles_metier.yaml`, alertes avec acquittement et notification, dashboard passant par l'API, correction du bug SHAP et du pilote PostgreSQL, horodatage des mesures, découpage temporel et références pour le RUL, validation de la règle d'alerte par rejeu temporel, infrastructure Docker (service `trainer`) et CI, tests (88 au total), mise à jour de la documentation.

**Arbitrages de l'équipe (réponses aux questions de l'IA) :**

| Question posée par l'IA | Réponse de l'équipe |
|---|---|
| Régénérer le dataset avec des défauts injectés (valeurs manquantes, capteurs figés, valeurs aberrantes) et ajouter une étape de nettoyage ? | « Oui, régénérer » |
| Supprimer du dépôt les fichiers de l'ancien pipeline (datasets publics, archive, ancien notebook) ? | « Oui, supprimer » |
| Quel format pour le support de soutenance ? | « on le fera nous meme le rapport » — le support de soutenance est rédigé par l'équipe, l'IA n'y a pas touché |

**Vérifications** : suite de tests exécutée après chaque étape ; contrôle de l'API sur les modèles réels (tous les endpoints, dont l'explication SHAP précédemment en erreur 500) ; bilan du nettoyage comparé aux défauts injectés ; plusieurs règles d'alerte comparées avant de retenir celle du fichier de configuration. Point non vérifié pendant la session : exécution complète de la stack `docker compose` (Docker Desktop arrêté).

**Adaptations notables** : la mesure par l'IA de 74 % d'alertes non suivies d'une panne sous 24 h a été approfondie avant conclusion ; l'analyse sur plusieurs horizons (90 % des alertes suivies d'une panne sous 7 jours) a conduit à présenter le besoin BM2 comme **non atteint à 24 h** et à proposer au client un horizon de 7 jours, plutôt qu'à modifier les données pour obtenir un meilleur chiffre.

### Prompt 7

> quand ta finis push toute les modification

**Production de l'IA** : commit et push des modifications sur le dépôt GitHub du projet à la fin des travaux.

### Ce qui reste à la charge de l'équipe

- Relire les modifications de code (diff Git) et être capable d'expliquer chaque choix en soutenance.
- Lancer `docker compose up --build` sur un clone neuf et vérifier le parcours complet (simulateur → alertes → dashboard).
- Rédiger le support de soutenance.

---

## 2. Autres usages de l'IA générative au cours du projet (à compléter par l'équipe)

Le cahier des charges cite comme usages possibles : description des écrans, règles métiers, scénarios d'usage, schéma de table, génération d'échantillons de données. Pour chaque usage antérieur à la session ci-dessus, compléter une fiche :

| Champ | À renseigner |
|---|---|
| **Outil** | (ex. ChatGPT, Claude, Copilot…) et version si connue |
| **Date** | |
| **Membre de l'équipe** | |
| **Prompt** | Texte exact saisi |
| **Production obtenue** | Résumé ou extrait de la réponse |
| **Analyse** | Ce qui a été jugé correct, incorrect ou incomplet, et comment cela a été vérifié |
| **Adaptation** | Ce qui a été modifié avant intégration |
| **Intégration** | Fichier ou livrable concerné (ex. `reference/generate_dataset.py`, `docs/GUIDE_UTILISATEUR_METIER.md`) |

### Fiche 1

| Champ | Contenu |
|---|---|
| Outil | |
| Date | |
| Membre de l'équipe | |
| Prompt | |
| Production obtenue | |
| Analyse | |
| Adaptation | |
| Intégration | |
