# Conduite du Changement - Déploiement de la Solution de Maintenance Prédictive

## 1. Contexte du changement

Le passage d'une maintenance **corrective/préventive planifiée** à une maintenance **prédictive basée sur l'IA** représente un changement significatif dans les pratiques de travail des équipes de production et de maintenance de MECHA.

Ce changement impacte :
- Les **méthodes de travail** : les techniciens doivent intégrer un nouvel outil dans leur routine quotidienne
- Les **processus de décision** : les priorités d'intervention sont désormais guidées par des indicateurs IA
- La **culture d'entreprise** : passage d'une logique réactive à une logique anticipatrice

---

## 2. Acteurs concernés

| Acteur | Rôle dans le changement | Impact |
|--------|------------------------|--------|
| **Direction générale** | Sponsor du projet, valide la stratégie | Faible (décisionnel) |
| **Direction industrielle** | Pilote le déploiement multi-sites | Moyen |
| **Responsables maintenance** (5 sites) | Utilisateurs principaux, adaptent la planification | **Fort** |
| **Techniciens maintenance** (~50 personnes) | Utilisateurs quotidiens du dashboard et des alertes | **Fort** |
| **Responsables production** | Intègrent les alertes dans la gestion des lignes | Moyen |
| **Opérateurs de production** | Informés des arrêts planifiés liés aux prédictions | Faible |
| **DSI** | Maintient l'infrastructure, support technique | Moyen |
| **Direction qualité** | Suit l'impact sur les taux de rebuts et non-conformités | Faible |

---

## 3. Principaux messages

### Pour la Direction
> "La solution signale à l'avance la majorité des pannes (3 sur 4 lors de la validation) et permet de planifier les interventions au lieu de les subir. Le pilote mesurera les arrêts réellement évités sur nos machines, pour chiffrer le retour sur investissement avec nos propres coûts d'arrêt."

### Pour les Responsables maintenance
> "Cet outil vous donne une visibilité anticipée sur l'état de vos machines. Il ne remplace pas votre expertise, il la renforce en vous fournissant des indicateurs objectifs pour prioriser vos interventions."

### Pour les Techniciens
> "Le dashboard vous alerte quand une machine reste à risque critique plusieurs heures de suite : c'est une machine à inspecter dans les prochains jours. Vous gardez la main sur la décision d'intervenir, et vos retours (acquittement, résultat d'inspection) servent à régler l'outil."

---

## 4. Démarche de participation

L'outil ne sera adopté que si les équipes terrain participent à son réglage. La démarche s'appuie sur trois leviers.

### 4.1 Utilisateurs clés et référents

- **Un référent maintenance par site** (5 au total), volontaire, formé en premier et associé à toutes les décisions de paramétrage de son site.
- **2 à 3 techniciens « utilisateurs clés »** par site pilote, qui testent l'outil en conditions réelles et remontent les irritants.
- Un **comité de pilotage** mensuel : direction industrielle, référents, DSI, équipe projet.

### 4.2 Co-construction des règles d'alerte pendant le pilote

Les seuils de risque, le nombre de relevés consécutifs qui déclenchent une alerte et les bornes physiques des capteurs sont regroupés dans un seul fichier lisible, `config/regles_metier.yaml`. Pendant le pilote :

- les référents et utilisateurs clés **choisissent ensemble** le compromis entre nombre d'alertes et pannes anticipées, à partir des mesures du rapport de validation (par exemple 2 ou 3 relevés consécutifs) ;
- ils arbitrent avec la direction industrielle l'**horizon de l'alerte** (24 h ou « à inspecter sous 7 jours ») ;
- chaque modification est tracée (version du fichier dans Git) et expliquée aux équipes.

### 4.3 Boucle de retour terrain

- **L'acquittement des alertes** dans le dashboard est la première boucle de retour : chaque alerte prise en charge est datée, et le résultat de l'inspection est noté dans la GMAO (panne confirmée, usure constatée, rien trouvé).
- **Ateliers de retour d'expérience** de 30 minutes, hebdomadaires pendant le pilote puis mensuels : revue des alertes de la période, des pannes non signalées et des propositions d'amélioration.
- Les retours alimentent le **ré-entraînement trimestriel** des modèles et l'ajustement du fichier de règles.

---

## 5. Plan de communication

| Quand | Qui (émetteur → public) | Message | Canal |
|---|---|---|---|
| Lancement du projet | Direction générale → tous les sites | Pourquoi la maintenance prédictive, objectifs, calendrier, rôle des équipes | Réunion d'information + note interne |
| Avant le pilote (S-2) | Direction industrielle → responsables maintenance et production | Périmètre du pilote, mode observation, appel aux volontaires utilisateurs clés | Réunion de site |
| Démarrage du pilote | Référent du site → techniciens | Ce que fait l'outil, comment lire et acquitter une alerte, ce que l'on attend de leurs retours | Atelier de prise en main + guide utilisateur |
| Pendant le pilote (chaque semaine) | Référent → équipe du site | Alertes de la semaine, pannes anticipées ou manquées, réglages décidés | Atelier de 30 min + affichage atelier |
| Fin du pilote | Comité de pilotage → direction et sites | Résultats mesurés, décision de passage en phase 2, ajustements retenus | Comité de pilotage + synthèse écrite |
| Extension (phases 2 et 3) | Référents des sites pilotes → nouveaux sites | Retour d'expérience entre pairs, bonnes pratiques | Visite inter-sites / visioconférence |
| En continu | DSI → utilisateurs | Évolutions, maintenances planifiées, incidents | Intranet (FAQ, actualités) |

---

## 6. Plan de formation

### 6.1 Formation des référents (Semaine 1-2)

| Session | Public | Durée | Contenu |
|---------|--------|-------|---------|
| Formation technique | 1 référent maintenance par site + DSI | 1 journée | Architecture, API, administration Docker, fichier de règles métier, dépannage |
| Formation fonctionnelle | Responsables maintenance (5 sites) | ½ journée | Les 5 pages du dashboard, interprétation et acquittement des alertes, processus de décision |

### 6.2 Formation des équipes (Semaine 3-4)

| Session | Public | Durée | Contenu |
|---------|--------|-------|---------|
| Prise en main | Techniciens maintenance (par groupes de 10) | 2 heures | Page Monitoring, lecture des niveaux de risque, traitement et acquittement d'une alerte, explication « pourquoi cette prédiction » |
| Sensibilisation | Responsables production | 1 heure | Comprendre les alertes, impact sur la planification production |

### 6.3 Accompagnement terrain (Semaine 5-8)

- Présence d'un référent formé sur chaque site pendant 2 semaines
- Hotline support (DSI) disponible pour les questions techniques
- Sessions de retour d'expérience hebdomadaires (30 min) pendant 1 mois

### 6.4 Supports de formation fournis

- Guide utilisateur métier (document livré avec la solution)
- Vidéos tutorielles courtes (2-3 min) pour chaque fonctionnalité du dashboard
- FAQ mise à jour en continu sur l'intranet

---

## 7. Stratégie de déploiement progressif

### Étape 1 : Pilote (Mois 1-2) — 1 site France

- Déploiement sur **un seul site** (le mieux équipé en capteurs IoT)
- Périmètre limité : **10-20 machines** représentatives
- Mode **observation** : les alertes sont générées mais n'impactent pas encore la planification officielle
- Objectif : valider les prédictions, calibrer le fichier de règles avec les utilisateurs clés (section 4.2), ré-entraîner les modèles sur les données réelles du site

### Étape 2 : Extension France (Mois 3-4) — 3 sites France

- Déploiement sur les **3 sites français**
- Mode **opérationnel** : les alertes sont intégrées dans le processus de planification maintenance
- Ajustements du fichier de règles si nécessaire en fonction des retours du pilote
- Mesure des premiers indicateurs de performance (réduction des arrêts non planifiés)

### Étape 3 : Déploiement international (Mois 5-6) — 5 sites

- Extension aux **2 sites espagnols**
- Adaptation linguistique du dashboard si nécessaire (interface multilingue)
- Homogénéisation des pratiques entre tous les sites

### Critères de passage d'une étape à la suivante

- Taux d'adoption du dashboard > 80% des techniciens du site
- Moins de 5 incidents techniques non résolus par semaine
- Retour positif des responsables maintenance sur la pertinence des alertes
- Chaque panne non signalée fait l'objet d'une analyse de cause partagée en atelier de retour

---

## 8. Gestion des résistances

| Résistance anticipée | Réponse |
|---------------------|---------|
| "L'IA va remplacer mon expertise" | L'outil est un assistant, pas un remplaçant. Le technicien garde la décision finale. |
| "Les prédictions ne sont pas fiables" | Phase pilote en mode observation ; résultats de validation présentés tels quels (3 pannes sur 4 signalées, 9 alertes sur 10 suivies d'une panne dans la semaine) ; chaque réglage est décidé avec les référents. |
| "Encore des fausses alertes" | Une alerte exige 3 relevés critiques consécutifs et ne se répète pas tant qu'elle n'est pas acquittée ; le volume (≈ 1 alerte par machine toutes les deux semaines) est suivi et ajusté en atelier. |
| "C'est un outil de plus à gérer" | Le dashboard est conçu pour être simple (5 pages, code couleur, une page Monitoring pour le quotidien). Formation courte de 2 h. |
| "Ça ne marchera pas sur nos machines spécifiques" | Les modèles sont ré-entraînés sur les données réelles du site pendant le pilote ; les règles d'alerte sont réglables par site. |

---

## 9. Indicateurs de succès du changement

| Indicateur | Cible à 6 mois |
|------------|----------------|
| Taux d'adoption du dashboard | > 85% des techniciens consultent le dashboard au moins 1 fois/jour |
| Réduction des arrêts non planifiés | -20% par rapport à la période précédente |
| Temps moyen de réaction à une alerte | < 4 heures |
| Satisfaction utilisateurs (enquête) | Note > 3.5/5 |
| Alertes suivies d'une panne ou d'une usure constatée | Cible à fixer avec le client selon l'horizon retenu. Référence de validation : 26 % sous 24 h, 90 % sous 7 jours. L'objectif initial (< 5 % de fausses alertes critiques à 24 h) n'est pas atteignable avec la version actuelle (cf. rapport de validation §4.4) |
| Pannes signalées à l'avance | ≥ 75 % (référence de validation : 75 % dans les 24 h précédant la panne) |
| Alertes acquittées sous 24 h | > 90 % |
| Participation aux ateliers de retour | ≥ 1 référent + 2 utilisateurs clés par site et par atelier |
