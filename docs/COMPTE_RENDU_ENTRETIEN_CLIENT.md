# Compte-Rendu des Échanges avec le Client

## Guide d'entretien simulé

**Contexte** : Entretien avec la Direction industrielle de MECHA pour recueillir les besoins métiers en vue de la conception d'une solution de maintenance prédictive basée sur l'IA.

**Date simulée** : Janvier 2025
**Participants** :
- M. Dupont — Directeur industriel, MECHA
- Mme Garcia — Responsable maintenance, site France (Lyon)
- M. Bernard — DSI, MECHA
- Équipe projet EPSI (4 apprenants)

---

### Questions posées et réponses collectées

**Q1 : Quelle est la problématique industrielle prioritaire que vous souhaitez adresser avec l'IA ?**

> *M. Dupont* : "Notre priorité, c'est de réduire les arrêts non planifiés. Quand une machine s'arrête en pleine production, on perd des heures, on prend du retard sur les commandes, et ça coûte très cher. On estime que les pannes imprévues nous coûtent environ 15% de notre capacité de production sur certains sites."

**Q2 : Comment fonctionne votre maintenance actuelle ?**

> *Mme Garcia* : "Aujourd'hui, on fait de la maintenance préventive planifiée : on intervient toutes les X heures de fonctionnement selon les recommandations constructeurs. Mais c'est soit trop tôt — on change des pièces encore fonctionnelles — soit trop tard — la machine tombe en panne avant l'intervention prévue. On fait aussi du correctif quand ça casse, bien sûr."

**Q3 : Quelles données avez-vous à disposition sur vos machines ?**

> *M. Bernard* : "On a déployé des capteurs IoT sur la plupart de nos machines : température, pression, vitesse de rotation, vibrations. Les données remontent toutes les heures dans notre système SCADA. On a aussi un historique de maintenance dans notre GMAO — les interventions, les pièces changées, les erreurs signalées. Par contre, les 5 sites ne sont pas tous au même niveau : les 2 sites espagnols ont été équipés plus récemment et ont moins d'historique."

**Q4 : Quelles machines sont les plus critiques ?**

> *Mme Garcia* : "Les centres d'usinage sont les plus critiques. Une panne sur un centre d'usinage peut bloquer toute une ligne. On a environ 100 machines de ce type réparties sur les 5 sites. Ce sont celles-là qu'on voudrait surveiller en priorité."

**Q5 : Quel horizon de prédiction serait utile pour vous ?**

> *M. Dupont* : "Si on pouvait savoir 24 heures à l'avance qu'une machine risque de tomber en panne, ça nous laisserait le temps de planifier une intervention sans impacter la production. L'idéal serait même d'avoir une estimation du temps restant avant la panne, pour prioriser."

**Q6 : Qui utilisera la solution au quotidien ?**

> *Mme Garcia* : "Les responsables maintenance de chaque site, et leurs équipes de techniciens. Il faut que ce soit simple : un tableau de bord avec des indicateurs visuels, des codes couleur, pas besoin de comprendre l'algorithme derrière. Et des alertes quand ça devient critique."

**Q7 : Avez-vous des contraintes techniques particulières ?**

> *M. Bernard* : "On veut que la solution soit conteneurisée pour faciliter le déploiement sur nos différents sites. On utilise Git pour le versioning. Et il faudra que ça s'intègre avec nos systèmes existants via des API. On a aussi des contraintes de sécurité : les données de production ne doivent pas sortir de notre réseau."

**Q8 : Quels seraient les critères de succès du projet pour vous ?**

> *M. Dupont* : "Concrètement, je veux voir un prototype qui fonctionne, avec des données réalistes, et qui me montre qu'on peut détecter une panne avant qu'elle arrive. Si le modèle a un bon taux de détection — disons 80% des pannes détectées — et pas trop de fausses alertes, c'est gagné. Et surtout, que les équipes terrain puissent comprendre et utiliser les résultats."

---

## Synthèse des besoins métiers collectés

### Besoins fonctionnels

| # | Besoin | Priorité | Source |
|---|--------|----------|--------|
| BF1 | Prédire les pannes machines 24h à l'avance | Haute | Direction industrielle |
| BF2 | Estimer le temps restant avant défaillance (RUL) | Haute | Direction industrielle |
| BF3 | Classifier l'état des machines (normal / à risque) | Haute | Resp. maintenance |
| BF4 | Afficher un tableau de bord visuel et intuitif | Haute | Resp. maintenance |
| BF5 | Générer des alertes par niveau de risque (code couleur) | Haute | Resp. maintenance |
| BF6 | Visualiser l'historique des données capteurs par machine | Moyenne | Ingénieurs fiabilité |
| BF7 | Comparer les performances des modèles IA | Moyenne | DSI |
| BF8 | Permettre la saisie manuelle de données capteurs pour test | Basse | Techniciens |

### Besoins techniques

| # | Besoin | Priorité | Source |
|---|--------|----------|--------|
| BT1 | Solution conteneurisée (Docker) | Haute | DSI |
| BT2 | API REST pour intégration avec les systèmes existants | Haute | DSI |
| BT3 | Code versionné sur Git | Haute | DSI |
| BT4 | CI/CD pour les tests et le build | Moyenne | DSI |
| BT5 | Exploiter les données capteurs IoT (temp., vibration, pression, vitesse) | Haute | DSI |
| BT6 | Données restent on-premise (pas de cloud externe) | Haute | DSI |

### Besoins métier / qualité

| # | Besoin | Priorité | Source |
|---|--------|----------|--------|
| BM1 | Recall > 80% (détecter au moins 80% des pannes) | Haute | Direction industrielle |
| BM2 | Taux de fausses alertes critiques < 5% | Haute | Resp. maintenance |
| BM3 | Solution utilisable sans compétence data science | Haute | Resp. maintenance |
| BM4 | Déploiement progressif site par site | Moyenne | Direction industrielle |

### Contraintes identifiées

- Hétérogénéité des équipements IoT entre les 5 sites
- Historique de données variable selon les sites (plus court en Espagne)
- Données déséquilibrées : les pannes sont rares (~3% des observations)
- Les techniciens n'ont pas de formation data : l'interface doit être très intuitive
- Confidentialité des données de production : pas de données réelles utilisables pour le PoC

### Correspondance besoins → solution livrée

Chiffres issus de `models/training_results.json` ; détail dans le [rapport de validation](RAPPORT_VALIDATION_DEPLOIEMENT.md).

| Besoin | Composant livré | Statut |
|--------|----------------|--------|
| BF1 - Prédiction 24h | Random Forest « panne dans 24 h » (`POST /predict/panne24h`) : F1 0,882 sur test 50/50, rappel 94 % sur la distribution réelle | Livré |
| BF2 - RUL | Random Forest Regressor (`POST /predict/rul`) : MAE 105 h en découpage temporel (modèle naïf : 224 h) — ordre de grandeur pour prioriser | Livré, précision limitée |
| BF3 - Classification normal / à risque | `POST /predict/panne` et `/predict/panne24h`, 4 niveaux de risque définis dans `config/regles_metier.yaml` ; détection d'anomalies (`/predict/anomalie`) | Livré |
| BF4 - Dashboard | Streamlit, 5 pages (dont Monitoring Temps Réel) ; prédictions via l'API ; explication de chaque prédiction | Livré |
| BF5 - Alertes | Règle « 3 relevés consécutifs au niveau critique », anti-doublon, acquittement, notification par webhook configurable (`GET /alertes`, `POST /alertes/{id}/acquitter`) | Livré |
| BF6 - Historique capteurs | Page « Analyse des capteurs » | Livré |
| BF7 - Comparaison des modèles | 6 modèles + modèle naïf, deux évaluations ; page « Performance des modèles » | Livré |
| BF8 - Saisie manuelle | Saisie manuelle et import CSV (`POST /predict/batch`) | Livré |
| BT1 - Docker | Dockerfile + docker-compose (5 services : postgres, trainer, api, dashboard, simulator) | Livré — exécution complète à revérifier avant la soutenance |
| BT2 - API | FastAPI, 15 endpoints, documentation Swagger (`/docs`), webhook d'ingestion `POST /webhook/sensor` | Livré |
| BT3 - Git | Dépôt GitHub structuré | Livré |
| BT4 - CI/CD | GitHub Actions : 88 tests sur Python 3.11 et 3.12, construction et vérification de l'image Docker | Livré |
| BT5 - Capteurs IoT | Température, vibration, courant, pression, vitesse ; contrôle de plausibilité et nettoyage | Livré |
| BT6 - On-premise | Tous les services sont auto-hébergés (aucun service cloud externe) ; la notification pointe vers une URL interne configurable | Atteint |
| BM1 - Recall > 80 % | 94 % des relevés précédant une panne détectés (relevé par relevé). Par événement, avec la règle d'alerte retenue : 75 % des pannes signalées dans les 24 h qui précèdent (82 % avec 2 relevés consécutifs, pour 45 % d'alertes en plus) | Atteint relevé par relevé ; partiellement par événement — choix de règle à arbitrer |
| BM2 - Fausses alertes critiques < 5 % | 74 % des alertes ne sont pas suivies d'une panne **dans les 24 h** ; 10 % ne sont pas suivies d'une panne **dans les 7 jours** | **Non atteint à 24 h** — proposition : redéfinir l'alerte à 7 jours (cf. rapport §4.4) |
| BM3 - Sans compétence data science | Code couleur, recommandation textuelle, explication des 3 facteurs principaux, guide utilisateur | Atteint (à confirmer pendant le pilote) |
| BM4 - Déploiement progressif | Plan en 3 phases (pilote observation → France → Espagne) ; règles paramétrables par site | Prévu (plan de déploiement) |
